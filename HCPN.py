import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torch.nn import Parameter

import dgl

class HCPN(nn.Module):
    def __init__(self, data_dim, dim_proto, dim_cls, num_class, n_proto_alloc, n_AFE_a_alloc, n_AFE_r_alloc, n_AFE_a_select, n_AFE_r_select, atom_t, w_dr_intra, w_dr_inter, w_obj_shr, dr_dis, n_nbs_per_hop, devices):
        super(HCPN, self).__init__()
        self.d_prot_a = dim_proto
        self.d_prot_c = dim_cls
        self.AFE_attr = [Parameter(torch.empty(data_dim, dim_proto, device=devices).uniform_(-np.sqrt(1./data_dim), np.sqrt(1./data_dim))) for i in range(n_AFE_a_alloc)]
        self.AFE_rela = [Parameter(torch.empty(data_dim, dim_proto, device=devices).uniform_(-np.sqrt(1. / data_dim), np.sqrt(1. / data_dim))) for i in range(n_AFE_r_alloc)]
        self.w_atom_r = Parameter(torch.tensor(0.0, device=devices), requires_grad=False) # weight for regulation contribution ratio within pair features
        self.n_AFE_a_alloc = n_AFE_a_alloc # how many to select from all embs
        self.n_AFE_r_alloc = n_AFE_r_alloc
        self.n_AFE_a_select = n_AFE_a_select
        self.n_AFE_r_select = n_AFE_r_select
        self.c_AFE_attr_id = 0 # from which emb matrix is being used currently
        self.c_AFE_rela_id = 0
        self.c_AFE_attr_id_end = None
        self.c_AFE_rela_id_end = None
        self.AFE_attr_id_rec = [0,n_AFE_a_alloc]
        self.AFE_rela_id_rec = [0,1]
        self.div_reg_t = 0.9 # threshold for forcing emb matrices to be orthogonal
        self.prototypes = utils.Component_prototypes(dim_proto, dim_cls, n_proto_alloc, n_AFE_a_select, n_AFE_r_select, [n_AFE_a_alloc, n_AFE_r_alloc], n_nbs_per_hop)
        self.classifier_simp_atom = nn.Linear(dim_proto*(n_AFE_a_select+n_AFE_r_select*sum(n_nbs_per_hop)), num_class) # a simplified classifier
        self.classifier_simp_emb = nn.Linear(dim_proto, num_class)  # a simplified classifier
        self.classifier_simp_obj = nn.Linear(dim_proto, num_class)
        self.classifier_simp_ao = nn.Linear(dim_proto*(n_AFE_a_select+n_AFE_r_select* sum(n_nbs_per_hop)+1), num_class)
        self.classifier_simp_aoc = nn.Linear(dim_proto*(n_AFE_a_select+n_AFE_r_select* sum(n_nbs_per_hop)+1)+self.d_prot_c, num_class)
        self.classifier_atten_GAT = utils.atten_classifier_GAT_o(dim_proto, dim_proto, num_class, inte='concat')
        self.num_class = num_class
        self.criterion = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss() cls_criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion_graph = torch.nn.BCEWithLogitsLoss()
        self.emb_pro_dis_loss = nn.MSELoss()
        self.atom_t = atom_t
        self.w_dr_intra = w_dr_intra # scaling factor of penalty on atom diversity
        self.w_dr_inter = w_dr_inter
        self.w_obj_shr = w_obj_shr # scaling factor of object shrink loss
        self.dr_dis = dr_dis # min distance used in diversity penalty
        self.sigmoid = nn.Sigmoid()
        self.batch_norm = nn.BatchNorm1d(dim_proto)
        self.relu = nn.ReLU()
        self.device = devices
        self.data_dim = data_dim
        self.ids_record = []
        self.proto_ids_record = []
        self.n_nbs_per_hop = n_nbs_per_hop
        self.AFE_a_ids_selected = None
        self.AFE_r_ids_selected = None
        self.task_AFE_map_a = {}
        self.task_AFE_map_r = {}
        self.n_AFE_a_ids_used = []
        self.n_AFE_r_ids_used = []

    def forward(self, data, c_ids, est_proto, task_id, reselect_AFE = False, est_obj=False, proto_cls=False):
        # data prepare
        train_ids, valida_ids, test_ids, _, multi_nbs, features, y_train, y_val, y_test, labels = data
        c_labels = labels[c_ids]
        nb_ids = [multi_nbs[id] for id in c_ids]
        nei_ids_sampled = utils.lil_sample(nb_ids, self.n_nbs_per_hop, flatten=True)
        nei_ids_sampled = np.array(nei_ids_sampled).reshape(-1)

        c_feats = torch.tensor(features[c_ids], dtype=torch.float, device=self.device)  # the features of current nodes [batch_size, dim_feats]
        batch_size = c_feats.shape[0]
        c_feats = c_feats.view(batch_size, 1, -1)

        nei_feats = torch.tensor(features[nei_ids_sampled], device=self.device, dtype = torch.float).view(batch_size, sum(self.n_nbs_per_hop), -1)

        pair_feats = self.w_atom_r*c_feats.repeat([1, sum(self.n_nbs_per_hop), 1]) + (1.0-self.w_atom_r)*nei_feats

        ## AFE selection
        if task_id not in self.task_AFE_map_a.keys():
            emb_attr_try = [(c_feats.view(batch_size, -1).mm(AFE)).view(batch_size, self.d_prot_a) for AFE in
                            self.AFE_attr]  # la * [batch, 1, d_proto]
            emb_rela_try = [(pair_feats.view(batch_size * sum(self.n_nbs_per_hop), -1).mm(AFE)).view(batch_size, -1, self.d_prot_a) for AFE in self.AFE_rela]  # n_emb_r * [batch, lr, d_proto]
            self.AFE_a_ids_selected, self.AFE_r_ids_selected = self.prototypes.AFE_select(emb_attr_try, emb_rela_try,
                                                                                          self.n_AFE_a_select,
                                                                                          self.n_AFE_r_select)
            if len(self.n_AFE_a_ids_used) + self.n_AFE_a_select <= self.n_AFE_a_alloc:
                self.AFE_a_ids_selected = list(
                    range(len(self.n_AFE_a_ids_used), len(self.n_AFE_a_ids_used) + self.n_AFE_a_select))
                self.n_AFE_a_ids_used.extend(self.AFE_a_ids_selected)
            if len(self.n_AFE_r_ids_used) + self.n_AFE_r_select <= self.n_AFE_r_alloc:
                self.AFE_r_ids_selected = list(range(len(self.n_AFE_r_ids_used),
                                                     len(self.n_AFE_r_ids_used) + self.n_AFE_r_select))  # [len(self.n_AFE_r_ids_used)]
                self.n_AFE_r_ids_used.extend(self.AFE_r_ids_selected)
            self.task_AFE_map_a[task_id], self.task_AFE_map_r[
                task_id] = self.AFE_a_ids_selected, self.AFE_r_ids_selected
        else:
            self.AFE_a_ids_selected, self.AFE_r_ids_selected = self.task_AFE_map_a[task_id], self.task_AFE_map_r[
                task_id]

        AFE_attr_selected, AFE_rela_selected = [self.AFE_attr[i] for i in self.AFE_a_ids_selected], [self.AFE_rela[i] for i in self.AFE_r_ids_selected]
        emb_attr = [(c_feats.view(batch_size, -1).mm(AFE)).view(batch_size, -1, self.d_prot_a) for AFE in AFE_attr_selected] # n_AFE_a * [batch, 1, d_proto]
        emb_rela = [(pair_feats.view(batch_size*sum(self.n_nbs_per_hop), -1).mm(AFE)).view(batch_size, -1, self.d_prot_a) for AFE in AFE_rela_selected] # n_AFE_r * [batch, n_nbs, d_proto]
        emb_attr_cat = torch.cat(emb_attr, dim=1) # [batch, n_AFE_a, d_proto]
        emb_attr_rela = torch.cat(emb_rela, dim=1)  # [batch, n_AFE_r*n_nbs, d_proto]
        atom_embs = torch.cat([emb_attr_cat, emb_attr_rela], dim=1)  # [batch, n_AFE_a + n_AFE_r*n_nbs, d_proto]

        atom_embs_n = F.normalize(atom_embs, p=2, dim=-1) # normalize each component embedding into a unit ball

        ## prototype interaction
        self.prototypes = self.prototypes.cuda(atom_embs_n.get_device())
        associated_atoms, associated_obj_embs, associated_cls, hard_corres_atom, hard_corres_obj = self.prototypes.update(c_ids, atom_embs_n, self.AFE_a_ids_selected, self.AFE_r_ids_selected, self.atom_t, est_proto) # [batch_size * (la+lr), num_protos] # correspondence.mm(self.prototypes.atoms[0:n_atoms]) # [batch_size * (la+lr), dim_proto]
        id_batch = torch.tensor(range(batch_size)).view(batch_size, 1, 1)
        id_dim = torch.tensor(range(self.d_prot_a)).view(1, 1, self.d_prot_a)
        self.atom_embs = atom_embs_n # [id_batch, selected_sorted_ids, id_dim]

        ## classifier
        c_labels = torch.tensor([np.argmax(label) for label in c_labels], dtype=torch.long, device=atom_embs_n.get_device())

        loss_emb_ato_dis = self.emb_pro_dis_loss(associated_atoms, self.atom_embs)

        # atomic embedding classification
        preds_emb = self.classifier_simp_atom(self.atom_embs.view(batch_size, self.d_prot_a * (self.n_AFE_r_select * sum(self.n_nbs_per_hop)+ self.n_AFE_a_select)))
        preds_emb = F.softmax(preds_emb, dim=1)
        loss_cls_emb = self.criterion(preds_emb, c_labels)
        # atom proto classification
        preds_atom = self.classifier_simp_atom(associated_atoms.view(batch_size, self.d_prot_a * (self.n_AFE_r_select * sum(self.n_nbs_per_hop)+ self.n_AFE_a_select)))
        preds_atom = F.softmax(preds_atom, dim=1)
        loss_cls_atom = self.criterion(preds_atom, c_labels)

        # obj&atom co-classification (concat)
        associated_cls = associated_cls.view(batch_size, self.d_prot_c)
        associated_obj = associated_obj_embs.view(batch_size, self.d_prot_a)
        associated_atoms = associated_atoms.view(batch_size, (self.n_AFE_r_select* sum(self.n_nbs_per_hop) + self.n_AFE_a_select)*self.d_prot_a)
        associated_aos = torch.cat([associated_atoms, associated_obj], dim=1)
        associated_aocs = torch.cat([associated_atoms, associated_obj, associated_cls], dim=1)
        preds_aoc = self.classifier_simp_aoc(associated_aocs.view(batch_size, self.d_prot_a * (1 + self.n_AFE_r_select* sum(self.n_nbs_per_hop) + self.n_AFE_a_select)+self.d_prot_c))
        preds_ao = self.classifier_simp_ao(associated_aos.view(batch_size, self.d_prot_a * (1 + self.n_AFE_r_select* sum(self.n_nbs_per_hop) + self.n_AFE_a_select)))
        preds_aoc = F.softmax(preds_aoc, dim=1)
        preds_ao = F.softmax(preds_ao, dim=1)
        loss_cls_ao = self.criterion(preds_ao, c_labels)
        loss_cls_aoc = self.criterion(preds_aoc, c_labels)

        if self.training and not est_proto:
            loss_cls_atom = torch.tensor(0., device = self.device)
            loss_cls_aoc = torch.tensor(0., device = self.device)
            loss_cls_ao = torch.tensor(0., device = self.device)
            #loss_emb_ato_dis = loss_emb_ato_dis*0
            loss_emb_ato_dis = torch.tensor(0., device = self.device)
        elif self.training and est_proto:
            loss_cls_atom = torch.tensor(0., device = self.device)
            #loss_cls_emb = torch.tensor(0., device = self.device)
            loss_emb_ato_dis = loss_emb_ato_dis*1

        # loss computation
        diver_reg_attr = torch.tensor([], device=self.device)
        diver_reg_rela = torch.tensor([], device=self.device)
        l_rec_attr = len(self.AFE_attr_id_rec)
        l_rec_rela = len(self.AFE_rela_id_rec)
        for i in range(l_rec_attr-2):
            m1 = F.normalize(torch.cat(self.AFE_attr[self.AFE_attr_id_rec[i]:self.AFE_attr_id_rec[i+1]], dim=1), p=2, dim=0)
            for j in range(i+1, l_rec_attr-1, 1):
                m2 = F.normalize(torch.cat(self.AFE_attr[self.AFE_attr_id_rec[j]:self.AFE_attr_id_rec[j+1]], dim=1), p=2, dim=0)
                cos_dis = m1.transpose(1,0).mm(m2)
                mask = (cos_dis>self.div_reg_t).float()
                cos_dis_triu = torch.triu(cos_dis * mask, diagonal=-1)
                diver_reg_attr = torch.cat((diver_reg_attr, cos_dis_triu))
        for i in range(l_rec_rela-2):
            m1 = F.normalize(torch.cat(self.AFE_rela[self.AFE_rela_id_rec[i]:self.AFE_rela_id_rec[i+1]], dim=1), p=2, dim=0)
            for j in range(i+1, l_rec_rela-1, 1):
                m2 = F.normalize(torch.cat(self.AFE_rela[self.AFE_rela_id_rec[j]:self.AFE_rela_id_rec[j+1]], dim=1), p=2, dim=0)
                cos_dis = m1.transpose(1,0).mm(m2)
                mask = (cos_dis > self.div_reg_t).float()
                diver_reg_rela = torch.cat((diver_reg_rela, torch.triu(cos_dis*mask, diagonal=-1)))

        diver_reg = 0

        for i in [diver_reg_attr.mean(),diver_reg_rela.mean()]:
            if torch.isnan(i)==0:
                diver_reg = diver_reg + i

        return [loss_cls_emb, loss_cls_atom, 10*loss_cls_aoc+loss_cls_ao, loss_emb_ato_dis, diver_reg], preds_emb, preds_atom, preds_ao, preds_aoc, self.atom_embs, associated_aocs

    def incre_AFE(self, tp, n, heritate = False):
        # increase the number of embedding matrices of the given type
        if tp == 'attr':
            for i in range(n):
                if heritate:
                    # if heritate from the most recemt AFE, just copy it
                    self.AFE_attr.append(Parameter(self.emb_attr[-1].detach().clone()))
                else:
                    self.AFE_attr.append(Parameter(torch.empty(self.data_dim, self.d_prot_a, device=self.device).uniform_(-np.sqrt(1./self.data_dim), np.sqrt(1./self.data_dim))))
            self.AFE_attr_id_rec.append(self.AFE_attr_id_rec[-1]+n)
            self.c_AFE_attr_id = self.AFE_attr_id_rec[-2]
            self.prototypes.atom_a_splits.append(self.prototypes.num_atoms)
            self.prototypes.AFE_attr_dict.append([0])
            print('self.AFE attr rec is increased to', self.AFE_attr_id_rec)
        elif tp == 'rela':
            for i in range(n):
                if heritate:
                    self.AFE_rela.append(Parameter(self.AFE_attr[-1].detach().clone()))
                else:
                    self.AFE_rela.append(Parameter(
                        torch.empty(self.data_dim, self.d_prot_a, device=self.device).uniform_(
                            -np.sqrt(1. / self.data_dim), np.sqrt(1. / self.data_dim))))
            self.AFE_rela_id_rec.append(self.AFE_rela_id_rec[-1] + n)
            self.c_AFE_rela_id = self.AFE_rela_id_rec[-2]
            self.prototypes.atom_r_splits.append(self.prototypes.num_atoms)
            self.prototypes.AFE_rela_dict.append([0])
            print('self.AFE rela rec is increased to',self.AFE_rela_id_rec)
