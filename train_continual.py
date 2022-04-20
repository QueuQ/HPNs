import argparse
import pickle
import shutil
import matplotlib.pyplot as plt


# torch
import torch.optim as optim

import HCPN
from utils import *

# settings
parser = argparse.ArgumentParser(description='Hierarchical Component Prototype Network')
parser.add_argument('--continual-cls', default=[[0, 1], [2, 3], [4, 5]])
parser.add_argument('--dim-proto', type=int, default=8, help='dimension of prototypes')
parser.add_argument('--dim-cls', type=int, default=8, help='dimension of class prototypes')
parser.add_argument('--base-lr', type=float, default=0.1, help='base learning rate')
parser.add_argument('--base-lr-adam', type=float, default=0.001, help='base learning rate for Adam')
parser.add_argument('--weight-decay', type=float, default=0.0001)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--test-batch-size', type=int, default=5000)
parser.add_argument('--nesterov', default=True)
parser.add_argument('--devices', type=int, default=0, help='indices for GPUs to use')
parser.add_argument('--num-epochs', type=int, default=90, help='number of training epochs')
parser.add_argument('--proto_est_epo', type=int, default=35, help='number of epoch before starting establishing prototypes, i.e. pre-train the embedding module')
parser.add_argument('--proto_cls_epo', type=int, default=50, help='num_epochs before training protos with classifi loss, only by emb proto distance loss')
parser.add_argument('--lr-decay', nargs='+', default=[35,35])
parser.add_argument('--lr-decay-proto', default=[], help='when to decay the learning rate for training prototype')
parser.add_argument('--shuffle_train_ids', default=False, help='whether to shuffle train data')
parser.add_argument('--fix_emb', default=False, help='whether to fix embedding module after starting establishing protos')
parser.add_argument('--e_fix_atom', default=[False, 40], help='whether to freeze atoms, and when to do this')
parser.add_argument('--train', default=True, help='whether to train the model')
parser.add_argument('--version_name', default='HCN_11', help='name of current version')
parser.add_argument('--n_nbs_per_hop', nargs='+', default=[1], help='#neighbors for different hops')
parser.add_argument('--n_AFE_a_alloc', type=int, default=22, help='number of attribute AFE allocated')
parser.add_argument('--n_AFE_r_alloc', type=int, default=22, help='number of relational AFE allocated')
parser.add_argument('--n_AFE_a_select', type=int, default=1, help='number of attribute embeddings to select from all embeddings')
parser.add_argument('--n_AFE_r_select', type=int, default=2, help='number of relation embeddings to select from all embeddings')
parser.add_argument('--n_atoms_alloc', nargs='+', default=[3000,3000,3000],
                    help='number of atom&obj&cls prototypes pre-defined')
parser.add_argument('--threshold_add_emb', default=10000.1,
                    help='if the norm of the grad of embedding matrices exceed threshold, add new embedding matrix')
parser.add_argument('--cool_down', default=200, help='after creating new embedding matrices, cool down for some epochs')
parser.add_argument('--data_path', default='./resources/datasets', help='path to cora and citeseer datasets')
parser.add_argument('--data_name', default='ogbn-products', help='name of the used dataset')
parser.add_argument('--atom_t', type=float, default=0.2, help='threshold for atom prototypes')
parser.add_argument('--w_dr_intra', default=0.0001,
                    help='weight on atom diversity loss for increasing distance between rows within each embedding matrix')
parser.add_argument('--w_dr_inter', default=0.0005,
                    help='weight on atom diversity loss for increasing distance between rows among different embedding matrices')
parser.add_argument('--w_obj_shr', default=0, help='weight on object shrink loss')
parser.add_argument('--dr_dis', default=1, help='min distance for atom diversity penalty')
parser.add_argument('--flatten', default=None, help='whether to merge multi-hop neighbor lists when generating them')
parser.add_argument('--optim', default=optim.SGD, help='can choose SGD or Adam')
parser.add_argument('--standard_data_split', default=False)
args = parser.parse_args()
args.cool_down = args.num_epochs
a = [int(i) for i in args.lr_decay]
args.lr_decay = a
a = [int(i) for i in args.n_atoms_alloc]
args.n_atoms_alloc = a
a = [int(i) for i in args.n_nbs_per_hop]
args.n_nbs_per_hop = a
#source = '/store/{}'.format(args.version_name)
optimizer = args.optim
if args.data_name in ['ogbn_arxiv','ogbn-arxiv']:
    args.continual_cls = [[35, 12],[15, 21],[28, 30], [16, 24], [10, 34], [8, 4], [5, 2], [27, 26], [36, 19], [23, 31], [9, 37], [13, 3], [20, 39], [22, 6], [38, 33], [25, 11], [18, 1], [14, 7], [0, 17], [29, 32]]  # descending order of class size, put difficult classes in the first
elif args.data_name in ['ogbn_products', 'ogbn-products']:
    args.continual_cls = [[4, 7], [6, 3], [12, 2], [0, 8], [1, 13], [16, 21], [9, 10], [18, 24], [17, 5], [11, 42],
                          [15, 20], [19, 23], [14, 25], [28, 29], [43, 22], [36, 44], [26, 37], [32, 31], [30, 27],
                          [34, 38], [41, 35], [39, 33],
                          [45, 40]]  # descending order of class size, the final class 46 with only 1 example is omitted
elif args.data_name in ['cora', 'citeseer']:
    args.continual_cls = [[0, 1], [2, 3], [4, 5]]
    #args.data_path = './resources/datasets'
else:
    args.continual_cls = [[2, 3], [0, 4]]
    #args.data_path = '/store/data/new_data'
acc_mat, acc_mat_final, grad_norms_attr, grad_norms_rela, previous_grad_norm_attr, previous_grad_norm_rela, attr_new_emb_epoc, rela_new_emb_epoc = [], [], [
    torch.tensor(0.)], [torch.tensor(0.)], 0., 0., args.cool_down, args.cool_down
load_model_path = './model.pt' # load and test with trained model without training from start

def train_batch(model, opt_AFE_lists, optimizer_model, optimizer_proto, data, epoch_id, proto_start, proto_cls_epo,
                fix_atom, batch_size=args.batch_size, schedulers=None):
    train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels = data
    model.train()
    loss_cls_emb_, loss_cls_atom_, loss_cls_obj_emb_, loss_ep_dis_, diver_reg, count, first_batch = 0, 0, 0, 0, 0, 0, True  # average classification loss

    if args.shuffle_train_ids:
        random.shuffle(train_ids)
    for batch_ids in chunks(train_ids, batch_size):
        loss_, _, _, _, _, _, _ = model(data, batch_ids, (epoch_id >= proto_start), task_id, proto_cls=(epoch_id >= proto_cls_epo))
        current_AFE_ids_a = model.task_AFE_map_a[task_id]
        current_AFE_ids_r = model.task_AFE_map_r[task_id]
        loss = sum(loss_)
        optimizer_model.zero_grad()
        optimizer_proto.zero_grad()
        for opts in opt_AFE_lists:
            for opt in opts:
                opt.zero_grad()

        loss.backward()
        optimizer_model.step()
        optimizer_proto.step()
        for opt in [opt_AFE_lists[0][i] for i in current_AFE_ids_a]:
            opt.step()
        for opt in [opt_AFE_lists[1][i] for i in current_AFE_ids_r]:
            opt.step()

        loss_cls_emb_ = loss_cls_emb_ + loss_[0].detach()
        loss_cls_atom_ = loss_cls_atom_ + loss_[1].detach()
        loss_cls_obj_emb_ = loss_cls_obj_emb_ + loss_[2].detach()
        loss_ep_dis_ = loss_ep_dis_ + loss_[3]  # .detach()
        diver_reg = diver_reg + loss_[4]
        count += 1

    loss_cls_emb = loss_cls_emb_ / count
    loss_cls_atom = loss_cls_atom_ / count
    loss_cls_obj_emb = loss_cls_obj_emb_ / count
    loss_ep_dis = loss_ep_dis_ / count
    return [loss_cls_emb, loss_cls_atom, loss_cls_obj_emb, loss_ep_dis, diver_reg], [grad_norms_attr, grad_norms_rela]


def test_batch(model, data, ids, batch_size=args.test_batch_size, save=False):
    train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels = data
    model.eval()
    pred_emb, pred_atom, pred_ao, pred_aoc = torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor(
        [])  # , dtype=torch.uint8)
    loss_t_cls, loss_t_ep_dis, count = 0, 0, 0
    for batch_ids in chunks(ids, batch_size):
        loss_, preds_emb, preds_atom, preds_ao, preds_aoc, _, associated_aocs = model(data, batch_ids, False, task_id, reselect_AFE = False)
        loss_t_cls = loss_[0] + loss_t_cls
        loss_t_ep_dis = loss_[3] + loss_t_ep_dis
        count += 1
        pred_emb = torch.cat((pred_emb, preds_emb.cpu()))
        pred_atom = torch.cat((pred_atom, preds_atom.cpu()))
        pred_ao = torch.cat((pred_ao, preds_ao.cpu()))
        pred_aoc = torch.cat((pred_aoc, preds_aoc.cpu()))

    ground_truth = torch.tensor([labels[i] for i in ids])

    loss_t_cls = loss_t_cls / count
    loss_t_ep_dis = loss_t_ep_dis / count

    acc_emb = torch.mean((torch.argmax(ground_truth, 1) == torch.argmax(pred_emb, 1)).float())
    acc_atom = torch.mean((torch.argmax(ground_truth, 1) == torch.argmax(pred_atom, 1)).float())
    acc_ao = torch.mean((torch.argmax(ground_truth, 1) == torch.argmax(pred_ao, 1)).float())
    acc_aoc = torch.mean((torch.argmax(ground_truth, 1) == torch.argmax(pred_aoc, 1)).float())
    return [loss_t_cls,
            loss_t_ep_dis], acc_emb, acc_atom, acc_ao, acc_aoc, model.prototypes.num_atoms, model.prototypes.num_objs


def process(classes, train, model=None, opts=None, schedulers=None, save_code_path=None, save_model_path=None,
            load_path=None, save_global_path=None):
    # each process runs one task
    # load data
    print('task id', task_id)
    data=None
    if args.data_name[0:3] == 'ogb':
        try:
            data = pickle.load(open('./resources/datasets/{}/preprocessed/{}_{}_.pkl'.format(args.data_name, args.continual_cls[task_id][0],
                                                        args.continual_cls[task_id][1]), 'rb'))
        except:
            print('no preprocessed data.')
    if data is None:
        print('beginning of loading data, {}'.format(time.time()))
        data = load_data(args.data_path, args.data_name, classes, len(args.n_nbs_per_hop), args.flatten, args=args)
        print('ending of loading data, {}'.format(time.time()))
    train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels = data

    if args.data_name[0:3] != 'ogb':
        train_ids = [i for i in train_ids if len(graph[i]) > 0]
        for i in test_ids:
            #print(graph[i])
            if len(graph[i]) == 0:
                graph[i] = i
                mnbs = []
                for j in range(len(args.n_nbs_per_hop)):
                    mnbs.append([i])
                multi_nbs[i] = mnbs
        # pickle.dump(data, open(f'/store/data/{args.data_name}/preprocessed/standard_split.pkl','wb'))
        data = [train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels]
    num_classes = len(classes)
    data_dim = features.shape[1]

    # load model
    device = args.devices[0] if type(args.devices) is list else args.devices

    if load_path is not None and not train:
        print('loading stored model {}'.format(load_path))
        model = torch.load(load_path)
        for p in model.parameters():
            p.requires_grad = True
    elif load_path is None and not train:
        print('still training, not loading any model')
    elif load_path is not None and train:
        print('still training, not loading any model')
    else:
        model = HCPN.HCPN(data_dim, args.dim_proto, args.dim_cls, num_classes, args.n_atoms_alloc,
                          args.n_AFE_a_alloc,
                          args.n_AFE_r_alloc, args.n_AFE_a_select, args.n_AFE_r_select, args.atom_t, args.w_dr_intra,
                          args.w_dr_inter, args.w_obj_shr,
                          args.dr_dis,
                          args.n_nbs_per_hop, args.devices).cuda(device)

    model.c_AFE_attr_id, model.c_AFE_rela_id = 0, 0

    # optimizer
    if opts == None:
        opt_AFE_attr_list = [optim.SGD([p],
                                       lr=args.base_lr,
                                       momentum=0.9,
                                       nesterov=args.nesterov,
                                       weight_decay=args.weight_decay) for p in model.AFE_attr]
        opt_AFE_rela_list = [optim.SGD([p],
                                       lr=args.base_lr,
                                       momentum=0.9,
                                       nesterov=args.nesterov,
                                       weight_decay=args.weight_decay) for p in model.AFE_rela]
        scheduler_AFE_attr_lsit = [optim.lr_scheduler.MultiStepLR(opt, args.lr_decay, gamma=0.1) for opt in
                                   opt_AFE_attr_list]
        scheduler_AFE_rela_lsit = [optim.lr_scheduler.MultiStepLR(opt, args.lr_decay, gamma=0.1) for opt in
                                   opt_AFE_rela_list]
        model_params = []
        proto_param_names = ['prototypes.{}'.format(param[0]) for param in model.prototypes.named_parameters()]
        # mdoel_param_names = ['prototypes.{}'.format(param[0]) for param in model.named_parameters()]
        for name, param in model.named_parameters():
            if name not in proto_param_names:
                model_params.append(param)
        optimizer_model = optim.SGD(model_params,
                                    lr=args.base_lr,
                                    momentum=0.9,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
        scheduler_model = optim.lr_scheduler.MultiStepLR(optimizer_model, args.lr_decay, gamma=0.1)

        optimizer_proto = optim.SGD(model.prototypes.parameters(),
                                    lr=args.base_lr,
                                    momentum=0.9,
                                    nesterov=args.nesterov,
                                    weight_decay=args.weight_decay)
        scheduler_proto = optim.lr_scheduler.MultiStepLR(optimizer_proto, args.lr_decay_proto, gamma=0.1)
        opts = [opt_AFE_attr_list, opt_AFE_rela_list, optimizer_model, optimizer_proto]
        schedulers = [scheduler_AFE_attr_lsit, scheduler_AFE_rela_lsit, scheduler_model, scheduler_proto]
    else:
        opt_AFE_attr_list, opt_AFE_rela_list, optimizer_model, optimizer_proto = opts
        scheduler_AFE_attr_lsit, scheduler_AFE_rela_lsit, scheduler_model, scheduler_proto = schedulers

    s = time.time()
    acc_aoc, acc_ao, accs_emb_test, tr_los_cls_emb, train_loss_ep_dis, test_loss_cls, test_loss_ep_dis, test_acc, train_acc = [], [], [], [], [], [], [], [], []
    if train:
        for i in range(args.num_epochs):
            t3 = time.time()
            # adjust learning rate
            if args.fix_emb:
                if i == args.proto_est_epo:
                    for p in list(model.AFE_rela.parameters()) + list(model.AFE_attr.parameters()):
                        p.requires_grad = False
            if args.e_fix_atom[0]:
                if i == args.e_fix_atom[1]:
                    model.prototypes.atoms.requires_grad = False

            s_train = time.time()
            loss_train, grad_norm = train_batch(model, [opt_AFE_attr_list, opt_AFE_rela_list], optimizer_model,
                                                optimizer_proto, data, i, args.proto_est_epo, args.proto_cls_epo,
                                                args.e_fix_atom,
                                                schedulers=[scheduler_AFE_attr_lsit, scheduler_AFE_rela_lsit])
            # grad_norms = grad_norms+grad_norm
            e_train = time.time()
            tr_los_cls_emb.append(loss_train[0])
            train_loss_ep_dis.append(loss_train[3])

            with torch.no_grad():
                loss_tr, acc_emb_train, acc_atom_train, acc_ao_train, acc_aoc_train, num_atoms, num_objs = test_batch(
                    model, data,
                    train_ids)
                train_acc.append(acc_atom_train)
                t1 = time.time()
                loss_t, acc_emb_test, acc_atom_test, acc_ao_test, acc_aoc_test, num_atoms, num_objs = test_batch(model,
                                                                                                                 data,
                                                                                                                 test_ids)
                accs_emb_test.append(acc_emb_test)
                test_acc.append(acc_atom_test)
                acc_ao.append(acc_ao_test)
                acc_aoc.append(acc_aoc_test)
                t2 = time.time()
                print(
                    'e{} los cls [emb_tr {:.2f} ato_tr {:.2f} obj_emb_tr {:.2f} emb_te {:.2f}] tr acc [emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.3f}] te acc[emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.3f}] #[ato {} obj {} cls {}]'.
                        format(i, loss_train[0], loss_train[1], loss_train[2], loss_t[0], acc_emb_train, acc_atom_train,
                               acc_ao_train, acc_aoc_train, acc_emb_test, acc_atom_test,
                               acc_ao_test, acc_aoc_test, num_atoms, num_objs, model.prototypes.num_cls))

            scheduler_model.step()
            scheduler_proto.step()
            for sch in scheduler_AFE_attr_lsit:
                sch.step()
            for sch in scheduler_AFE_rela_lsit:
                sch.step()

            t4 = time.time()
            # print('time consumed for one epoch is', t4-t3)
            if save_code_path is not None:
                lr = scheduler_model.optimizer.param_groups[0]['lr']
                with open(save_code_path + '/description.txt', 'a') as f:
                    f.write(
                        '\n e{} los cls [emb_tr {:.2f} ato_tr {:.2f} obj_emb_tr {:.2f} emb_te {:.2f}] tr acc [emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.3f}] te acc[emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.3f}] #[ato {} obj {} cls {}]'.
                            format(i, loss_train[0], loss_train[1], loss_train[2], loss_t[0], acc_emb_train,
                                   acc_atom_train, acc_ao_train, acc_aoc_train, acc_emb_test, acc_atom_test,
                                   acc_ao_test, acc_aoc_test, num_atoms, num_objs, model.prototypes.num_cls))

        if save_model_path is not None:
            print('saving model to', save_model_path)
            torch.save(model, save_model_path)
        if save_global_path is not None:
            torch.save(model, save_global_path)

        e = time.time()
        print('time consumed for training is', e - s)
        mean_epos = 10
        print(
            'over last {} epochs, average acc emb test is {:.5f}, acc atom test {:.5f}, acc_ao_test {:.5f}, acc_aoc_test {:.5f}'.format(
                mean_epos,
                np.mean(accs_emb_test[-mean_epos:]), np.mean(test_acc[-mean_epos:]), np.mean(acc_ao[-mean_epos:]),
                np.mean(acc_aoc[-mean_epos:])))
        if save_code_path is not None:
            with open(save_code_path + '/description.txt', 'a') as f:
                f.write(
                    '\n over last {} epochs, average acc emb test is {:.5f}, acc atom test {:.5f}, acc_ao_test {:.5f}, acc_aoc_test {:.5f}'.format(
                        mean_epos,
                        np.mean(accs_emb_test[-mean_epos:]), np.mean(test_acc[-mean_epos:]),
                        np.mean(acc_ao[-mean_epos:]), np.mean(acc_aoc[-mean_epos:])))
        acc_aoc_test_result = np.mean(acc_aoc[-mean_epos:])
    else:
        with torch.no_grad():
            loss_tr, acc_emb_train, acc_atom_train, acc_ao_train, acc_aoc_train, num_atoms, num_objs = test_batch(model,
                                                                                                                  data,
                                                                                                                  train_ids,
                                                                                                                  save=classes)
            loss_t, acc_emb_test, acc_atom_test, acc_ao_test, acc_aoc_test, num_atoms, num_objs = test_batch(model,
                                                                                                             data,
                                                                                                             test_ids,
                                                                                                             save=classes)
            acc_aoc_test_result = acc_aoc_test

        print(
            'los cls [emb_tr {:.2f} emb_te {:.2f}] ep dis:[tr {:.3f} te {:.3f}] tr acc [emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.3f}] te acc[emb {:.3f} ato {:.3f} ao {:.3f} aoc {:.4f}] #[ato {} obj {} cls {}]'.
                format(loss_tr[0], loss_t[0], loss_tr[-1], loss_t[-1], acc_emb_train, acc_atom_train, acc_ao_train,
                       acc_aoc_train, acc_emb_test, acc_atom_test, acc_ao_test, acc_aoc_test, num_atoms, num_objs,
                       model.prototypes.num_cls))
        if save_code_path is not None:
            with open(save_code_path + '/description.txt', 'a') as f:
                f.write(
                    '\n los cls [emb_tr {:.2f} emb_te {:.2f}] ep dis:[tr {:.3f} te {:.3f}] tr acc [emb {:.3f} ato {:.3f} aoc {:.3f}] te acc[emb {:.3f} ato {:.3f} aoc {:.3f}] #[ato {} obj {} cls {}]'.
                        format(loss_tr[0], loss_t[0], loss_tr[-1], loss_t[-1], acc_emb_train, acc_atom_train,
                               acc_aoc_train, acc_emb_test, acc_atom_test, acc_aoc_test, num_atoms, num_objs,
                               model.prototypes.num_cls))
    return model, opts, schedulers, acc_aoc_test_result


with torch.autograd.set_detect_anomaly(True):
    print('type of --n_atoms_alloc',type(args.n_atoms_alloc[0]))
    if args.train:
        descript = 'n'#input('input a description or type n')
        num_tasks = len(args.continual_cls)
        continual_task_names = []
        c_task_name = ''
        log_folder_name = f'./log/{args.data_name}_{time.localtime(time.time())[0:5]}_continual'

        for task_id in range(num_tasks):
            c_cls_names = ''
            for cls in args.continual_cls[task_id]:
                c_cls_names = c_cls_names + str(cls)
            c_task_name = c_task_name + c_cls_names + '_'
            continual_task_names.append(c_task_name + '.pt')

        for task_id in range(num_tasks):
            # train continually on each task
            print('task {}'.format(args.continual_cls[task_id]))

            save_model_path = log_folder_name + f'/{continual_task_names[task_id]}'
            if task_id != 0:
                if log_folder_name is not None:
                    with open(log_folder_name + '/description.txt', 'a') as f:
                        f.write('\n task {}'.format(args.continual_cls[task_id]))
                load_model_path = log_folder_name + '/{}'.format(continual_task_names[task_id - 1])
                c = args.continual_cls[task_id]
                _, _, _, acc_aoc_test = process(classes=c, train=True, model=model, opts=opts,
                                                schedulers=schedulers, save_code_path=log_folder_name,
                                                save_model_path=save_model_path, load_path=load_model_path)
                acc_mat.append(acc_aoc_test)
            elif task_id == 0:
                mkdir_if_missing(log_folder_name)
                with open(log_folder_name + '/description.txt', 'w+') as f:
                    f.write(descript)
                    f.write('\n task {}'.format(args.continual_cls[task_id]))
                c = args.continual_cls[task_id]
                model, opts, schedulers, acc_aoc_test = process(classes=c, train=True,
                                                                save_code_path=log_folder_name,
                                                                save_model_path=save_model_path, load_path=None)
                acc_mat.append(acc_aoc_test)

        t = np.arange(0, len(grad_norms_rela), 1)
        plt.plot(t, grad_norms_rela, 'r--')
        plt.plot(t, grad_norms_attr, 'b--')
        plt.show()

        # test the last model on each task
        load_model_path = log_folder_name + '/{}.pt'.format(continual_task_names[-1])
        for task_id in range(num_tasks):
            print('task {}'.format(args.continual_cls[task_id]))
            _, _, _, acc_aoc_test = process(classes=args.continual_cls[task_id], train=False, model=model)
            acc_mat_final.append(acc_aoc_test.item())

        AM = torch.mean(torch.tensor(acc_mat_final))
        FM_ = torch.tensor([acc_mat_final[i] - acc_mat[i] for i in range(num_tasks)])
        FM = torch.mean(FM_)
        acc_final = ['{:.4f}'.format(acc) for acc in acc_mat_final]
        acc_prev = ['{:.4f}'.format(acc) for acc in acc_mat]
        print('AM: {}, FM: {}'.format(AM, FM))
        print(acc_final)
        print(acc_prev)
        if log_folder_name is not None:
            with open(log_folder_name + '/description.txt', 'a') as f:
                f.write('\n AM: {}, FM: {}'.format(AM, FM))
                f.write('\n acc_final {}'.format(acc_final))
                f.write('\n acc__prev {}'.format(acc_prev))

        #shutil.copytree(source, log_folder_name + '/code', ignore=shutil.ignore_patterns('log')) # for saving code

    elif not args.train:
        print('testing in continual setting')
        try:
            load_model_path
            print('loading stored model {}'.format(load_model_path))
            model = torch.load(load_model_path)
        except NameError:
            print('Not loading stored model')
            load_model_path = None
        num_tasks = len(args.continual_cls)
        continual_task_names = []
        c_task_name = ''
        log_folder_name = './{}/log/{}_{}_{}_{}_{}_continual'.format(args.version_name,
                                                                          time.localtime(time.time())[0],
                                                                          time.localtime(time.time())[1],
                                                                          time.localtime(time.time())[2],
                                                                          time.localtime(time.time())[3],
                                                                          time.localtime(time.time())[4])

        for task_id in range(num_tasks):
            c_cls_names = ''
            for cls in args.continual_cls[task_id]:
                c_cls_names = c_cls_names + str(cls)
            c_task_name = c_task_name + c_cls_names + '_'
            # load previous models and test
            c_model_path = load_model_path.split(c_task_name)[0] + '{}.pt'.format(c_task_name)
            c_model = torch.load(c_model_path)
            _, _, _, acc_aoc_test = process(classes=args.continual_cls[task_id], train=False, model=c_model)
            acc_mat.append(acc_aoc_test)
            continual_task_names.append(c_task_name + '.pt')

        t = np.arange(0, len(grad_norms_rela), 1)
        plt.plot(t, grad_norms_rela, 'r--')
        plt.plot(t, grad_norms_attr, 'b--')
        plt.show()

        # test the last model on each task
        for task_id in range(num_tasks):
            print('task {}'.format(args.continual_cls[task_id]))
            _, _, _, acc_aoc_test = process(classes=args.continual_cls[task_id], train=False, model=model)
            acc_mat_final.append(acc_aoc_test.item())

        AM = torch.mean(torch.tensor(acc_mat_final))
        FM_ = torch.tensor([acc_mat_final[i] - acc_mat[i] for i in range(num_tasks)])
        FM = torch.mean(FM_)
        acc_final = ['{:.4f}'.format(acc) for acc in acc_mat_final]
        acc_prev = ['{:.4f}'.format(acc) for acc in acc_mat]
        print('AM: {}, FM: {}'.format(AM, FM))
        print(acc_final)
        print(acc_prev)
        if log_folder_name is not None:
            with open(log_folder_name + '/description.txt', 'a') as f:
                f.write('\n AM: {}, FM: {}'.format(AM, FM))
                f.write('\n acc_final {}'.format(acc_final))
                f.write('\n acc__prev {}'.format(acc_prev))

        #shutil.copytree(source, log_folder_name + '/code', ignore=shutil.ignore_patterns('log'))

