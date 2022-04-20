import argparse
import pickle
import shutil
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import HCPN
from utils import *

# convert the data from OGB,
# Conversion
from ogb.nodeproppred import NodePropPredDataset
import numpy as np
import operator
datasets = {'ogbn-arxiv','ogbn-products'}

# settings
parser = argparse.ArgumentParser(description='Hierarchical Component Prototype Network')
parser.add_argument('--continual-cls', default=[[0, 1], [2, 3], [4, 5]])
parser.add_argument('--dim-proto', default=8, help='dimension of prototypes')
parser.add_argument('--dim-cls', default=8, help='dimension of class prototypes')
parser.add_argument('--base-lr', default=0.1, help='base learning rate')
parser.add_argument('--base-lr-adam', default=0.001, help='base learning rate for Adam')
parser.add_argument('--weight-decay', default=0.0001)
parser.add_argument('--batch-size', default=100)
parser.add_argument('--test-batch-size', default=5000)
parser.add_argument('--nesterov', default=True)
parser.add_argument('--devices', type=int, default=0, help='indices for GPUs to use')
parser.add_argument('--num-epochs', default=90, help='number of training epochs')
parser.add_argument('--proto_est_epo', default=35,
                    help='number of epoch before starting establishing prototypes, i.e. pre-train the embedding module')
parser.add_argument('--proto_cls_epo', default=50,
                    help='num_epochs before training protos with classifi loss, only by emb proto distance loss')
parser.add_argument('--lr-decay', default=[35, 35])
parser.add_argument('--lr-decay-proto', default=[], help='when to decay the learning rate for training prototype')
parser.add_argument('--shuffle_train_ids', default=False, help='whether to shuffle train data')
parser.add_argument('--fix_emb', default=False,
                    help='whether to fix embedding module after starting establishing protos')
parser.add_argument('--e_fix_atom', default=[False, 40], help='whether to freeze atoms, and when to do this')
parser.add_argument('--train', default=True, help='whether to train the model')
parser.add_argument('--version_name', default='HCN_11', help='name of current version')
parser.add_argument('--n_nbs_per_hop', default=[1], help='#neighbors for different hops')
parser.add_argument('--n_AFE_a_alloc', default=22, help='number of attribute AFE allocated')
parser.add_argument('--n_AFE_r_alloc', default=22, help='number of relational AFE allocated')
parser.add_argument('--n_AFE_a_select', default=1, help='number of attribute embeddings to select from all embeddings')
parser.add_argument('--n_AFE_r_select', default=2, help='number of relation embeddings to select from all embeddings')
parser.add_argument('--n_atoms_alloc', default=[8000, 8000, 2000],
                    help='number of atom&obj&cls prototypes pre-defined')
parser.add_argument('--threshold_add_emb', default=10000.1,
                    help='if the norm of the grad of embedding matrices exceed threshold, add new embedding matrix')
parser.add_argument('--cool_down', default=200, help='after creating new embedding matrices, cool down for some epochs')
parser.add_argument('--data_path', default='./resources/datasets', help='path to cora and citeseer datasets')
parser.add_argument('--data_name', default='ogbn-arxiv', help='name of the used dataset')
parser.add_argument('--atom_t', default=0.2, help='threshold for atom prototypes')
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
if args.data_name in ['ogbn_arxiv', 'ogbn-arxiv']:
    args.continual_cls = [[35, 12], [15, 21], [28, 30], [16, 24], [10, 34], [8, 4], [5, 2], [27, 26], [36, 19],
                          [23, 31], [9, 37], [13, 3], [20, 39], [22, 6], [38, 33], [25, 11], [18, 1], [14, 7], [0, 17],
                          [29, 32]]  # descending order of class size, put difficult classes in the first
elif args.data_name in ['ogbn_products','ogbn-products']:
    args.continual_cls = [[4, 7], [6, 3], [12, 2], [0, 8], [1, 13], [16, 21], [9, 10], [18, 24], [17, 5], [11, 42],
                          [15, 20], [19, 23], [14, 25], [28, 29], [43, 22], [36, 44], [26, 37], [32, 31], [30, 27],
                          [34, 38], [41, 35], [39, 33],
                          [45, 40]]  # descending order of class size, the final class 46 with only 1 example is omitted

def process(classes):
    # each process runs one task
    # load data
    print('task id', task_id)
    print('beginning of loading data, {}'.format(time.time()))
    data = load_data(args.data_path, args.data_name, classes, len(args.n_nbs_per_hop), args.flatten, args=args)
    print('ending of loading data, {}'.format(time.time()))
    train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels = data
    train_ids = [i for i in train_ids if len(graph[i]) > 0]
    for i in test_ids:
        #print(graph[i])
        if len(graph[i]) == 0:
            graph[i] = i
            mnbs = []
            for j in range(len(args.n_nbs_per_hop)):
                mnbs.append([i])
            multi_nbs[i] = mnbs
    data = [train_ids, valida_ids, test_ids, graph, multi_nbs, features, y_train, y_val, y_test, labels]
    pickle.dump(data, open(
        f'./resources/datasets/{args.data_name}/preprocessed/{args.continual_cls[task_id][0]}_{args.continual_cls[task_id][1]}_.pkl',
        'wb'))

with torch.autograd.set_detect_anomaly(True):
    # a = os.listdir(f'./resources/datasets')
    dataset = NodePropPredDataset(name=args.data_name)
    # dataset = NodePropPredDataset(name = 'ogbn-papers100M')
    # feat, edge, label = np.load('/store/data/ogbn_arxiv/node_feat.npy'), np.load('/store/data/ogbn_arxiv/edges.npy'), np.load('/store/data/ogbn_arxiv/labels.npy')
    node_feats = dataset.graph['node_feat']
    edge_feats = dataset.graph['edge_index']
    labels = dataset.labels
    mkdir_if_missing(f'./resources/datasets/{args.data_name}')
    np.save(f'./resources/datasets/{args.data_name}/node_feats.npy', node_feats)
    np.save(f'./resources/datasets/{args.data_name}/edge_feats.npy', edge_feats)
    np.save(f'./resources/datasets/{args.data_name}/labels.npy', labels)

    num_tasks = len(args.continual_cls)
    for task_id in range(num_tasks):
        print('task {}'.format(args.continual_cls[task_id]))
        c = args.continual_cls[task_id]
        process(classes=c)
