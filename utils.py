import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import json
import sys
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import random
import time
import os
import errno
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.graphproppred import DglGraphPropPredDataset, collate_dgl, Evaluator
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class classifier_lstm(nn.Module):
    def __init__(self, n_cls, d_atom, d_obj, d_cls = 0):
        super(classifier_lstm, self).__init__()
        self.lstm_oa = nn.LSTM(d_obj, d_atom)
        self.lstm_a_out = nn.LSTM(d_atom, d_atom)
        self.classifier = nn.Linear(d_atom, n_cls)

    def forward(self, atoms, objs):
        # atoms [batch_size, (lr+la) * dim], objs [batch_size, dim]
        batch_size = atoms.shape[0]
        objs = objs.view(1, batch_size, -1)
        atoms = atoms.view(1, batch_size, -1)
        out_obj, (hid_obj, cell_obj) = self.lstm_oa(objs)
        out_atom, (hid_atom, cell_atom) = self.lstm_a_out(atoms, (hid_obj, cell_obj))
        predictions = self.classifier(hid_atom)
        return predictions.squeeze()

def cos_dis_loss(a, b):
    # input [*, dim_vector]
    dim_vector = a.shape[-1]
    if b.shape[-1] != dim_vector:
        raise ValueError('given vectors with different lengths')
    return torch.sum(1 - a.view(-1, dim_vector).mm(b.view(-1, dim_vector).transpose(1,0)))

def lil_sample(lil, n_per_list, flatten = None):
    # sample list of lists
    results = []
    for l in lil:
        ss = []
        for ll in range(len(l)):
            if not flatten:
                ss.append(random.choices(l[ll], k=n_per_list[ll]))
            elif flatten:
                ss = ss+random.choices(l[ll], k=n_per_list[ll])
        results.append(ss)
    return results

def multi_nb_sample(c_ids, edge_lists, n_per_hop, flatten = None):
    # sample from multi-hop neighbors, each hop corresponds to a given number
    # n_per_hop denotes how many examples to sample for each hop
    # returns sampled multi-hop neighbors for each node [batch_size, n_hop]
    n_hop = len(n_per_hop)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(edge_lists))
    adj_m = adj
    adj_multi = [] # denote multi-hop connections

    # create multi-hop adjacency lists
    t1 = time.time()
    for h in range(n_hop):
        adj_multi.append((adj_m>0)*1)
        adj_m = adj_m.dot(adj)
    t6 = time.time()
    id_map = [k for k in edge_lists]

    t2 = time.time()
    for k in edge_lists:
        l = edge_lists[k]
        for n in l:
            if not n in id_map:
                id_map.append(n)
    t3 = time.time()
    id_map = np.array(id_map)
    adj_list_multi = [] # [batch_size, n_hop, *]
    t4 = time.time()
    for id in c_ids:
        mapped_id = np.where(id_map==id)[0][0]
        nbs_multi = []
        for h in range(n_hop):
            nbs = random.choices(adj_multi[h][mapped_id].indices, k=n_per_hop[h])
            if flatten:
                nbs_multi += [id_map[i] for i in nbs]
            elif not flatten:
                nbs_multi.append([id_map[i] for i in nbs])

        adj_list_multi.append(nbs_multi)
    t5 = time.time()
    #print('t1 to t6 is {}, t6 to t2 is {}, t2 to t3 is {}, t3 to t4 is {}, t4 to t5 is {}'.format(t6-t1, t2-t6, t3-t2, t4-t3, t5-t4))
    return adj_list_multi #, adj_multi

def multi_nb(edge_lists, n_hop, flatten = None):
    # genegrate multi-hop neighbors
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(edge_lists))
    adj_m = adj
    adj_multi = [] # denote multi-hop connections

    # create multi-hop adjacency lists
    for h in range(n_hop):
        adj_multi.append((adj_m>0)*1)
        adj_m = adj_m.dot(adj)
    adj_list_multi = [] # [batch_size, n_hop, *]
    for id in edge_lists.keys():
        nbs_multi = []
        for h in range(n_hop):
            nbs = adj_multi[h][id].indices.tolist()
            if flatten:
                nbs_multi += nbs
            elif not flatten:
                nbs_multi.append(nbs)

        adj_list_multi.append(nbs_multi)
    return adj_list_multi#, adj_multi

def multi_nb_seq(edge_lists, n_hop, flatten = None):
    # genegrate multi-hop neighbors via batches for large graphs, sequential version of multi-nb
    adj_list_multi = edge_lists
    for i in range(n_hop-1):
        for j in range(len(adj_list_multi)):
            #print(j)
            nbs_current_node = edge_lists[j]
            updated_nbs_current_node = set(nbs_current_node)
            for k in nbs_current_node:
                updated_nbs_current_node.update(edge_lists[k])
            adj_list_multi[j] = list(updated_nbs_current_node)

    return adj_list_multi#, adj_multi

def block_diag(mtr_size, block_size):
    # generate a block square diagonal matrix with equally sized blocks
    m = torch.zeros([mtr_size, mtr_size])
    if mtr_size%block_size:
        raise ValueError('matrix size cannot be divided evenly by block size')
    n_blocks = int(mtr_size/block_size)
    for i in range(n_blocks):
        m[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = 1
    return m

def chunks(lst, n, shuffle = True):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

class MLP_classifier(nn.Module):
    def __init__(self, channel_list):
        super(MLP_classifier, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.layer_list = nn.ModuleList([nn.Linear(in_channel, out_channel) for (in_channel, out_channel) in channel_list])

    def forward(self, input, labels):
        depth = len(self.layer_list)
        for layer in range(depth):
            input = self.layer_list[layer](input)

        #pred =
        loss = self.loss(input, labels)

        return loss

def onehot_encoding(length, ids):
    if type(ids) == int:
        # if only one index to convert
        encoding = np.zeros(length)
        encoding[ids] = 1
        return encoding
    else:
        output = np.zeros([len(ids), length])
        for idx in range(len(ids)):
            output[idx][ids[idx]] = 1
        return output

def binary_position_encoding(num_digits, ids):
    # convert an integer into a binary encoding. Each 10 positions ecode 1 digit
    if type(ids) == int:
        # if only one index to convert
        encoding = np.zeros([num_digits,10])
        id_ = (num_digits-len(str(ids)))*'0' + str(ids)
        for i in range(num_digits):
            position = int(id_[i])
            encoding[i][position] = 1
        return encoding.reshape(-1)
    else:
        output = []
        for idx in ids:
            encoding = np.zeros([num_digits, 10])
            id_ = (num_digits-len(str(idx)))*'0' + str(idx)
            for i in range(num_digits):
                position = int(id_[i])
                encoding[i][position] = 1
            output.append(encoding.reshape(-1))
        return np.array(output)

def load_data_part_G(data_path, dataset_str, class_ids, n_hop, flatten):
    #num_class = len(class_ids)
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}/{}/ind.{}.{}".format(data_path, dataset_str, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    num_class_total = y.shape[1]
    test_idx_reorder = parse_index_file("{}/{}/ind.{}.test.index".format(data_path, dataset_str, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    class_labels = []
    for c in class_ids:
        label = sample_mask(c, y.shape[1]) * 1
        class_labels.append(label)
    for k in graph:
        jm = (labels[k] == class_labels) # compare label with candidate classes
        jm = np.sum(jm, 1) # if label matches a candidate class exactly, num_class will be in jm after sum
        jm = (jm == num_class_total)
        if not np.any(jm):
            # if a node does not belong to current classes, then isolate it
            graph[k] = [k]
        else:
            to_pop = []
            for t in range(len(graph[k])):
                #print('t is', t)
                jm1 = (labels[graph[k][t]] == class_labels)  # compare label with candidate classes
                jm1 = np.sum(jm1, 1)  # if label matches a candidate class exactly, num_class will be in jm after sum
                jm1 = (jm1 == num_class_total)
                if not np.any(jm1):
                    # if a node connects to a neighbor not in current class, remove this neighbor
                    to_pop.append(graph[k][t])
            for p in to_pop:
                graph[k].remove(p)

    #adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    # idx_test = test_idx_range.tolist()
    # idx_val = range(len(y), len(y) + 500)
    # idx_train = range(len(y))
    idx_test_candidate = test_idx_range.tolist()
    idx_test = []
    idx_val_candidate = list(range(len(y), len(y) + 500))
    idx_val = []

    idx_train = []
    for label in class_labels:
        ids_selected = np.matmul(y, label).nonzero()[0].tolist()
        idx_train = idx_train + ids_selected
        for id_can in idx_test_candidate:
            if sum(labels[id_can] * label):
                idx_test.append(id_can)
        for id_can in idx_val_candidate:
            if sum(labels[id_can] * label):
                idx_val.append(id_can)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # select only the involved classes in the labels
    y_train = y_train[:,class_ids]
    y_val = y_val[:, class_ids]
    y_test = y_test[:, class_ids]
    labels = labels[:, class_ids]

    # return multi-hop lists
    multi_nbs = multi_nb(graph, n_hop, flatten)
    return idx_train, idx_val, idx_test, graph, multi_nbs, features.todense(), y_train, y_val, y_test, labels

def load_newdata_part_G(data_path, dataset_str, class_ids, n_hop, flatten):
    # load edges and build the graph
    graph_ = dict()
    with open("{}/{}/out1_graph_edges.txt".format(data_path, dataset_str), 'r') as f:
        edges_str = f.readlines()[1:]
        for edge in edges_str:
            source, target = int(edge.split('\t')[0]), int(edge.split('\t')[1])
            if source in graph_.keys():
                if target not in graph_[source]:
                    graph_[source].append(target)
            elif source not in graph_.keys():
                graph_[source] = [target]

            if target in graph_.keys():
                if source not in graph_[target]:
                    graph_[target].append(source)
            elif target not in graph_.keys():
                graph_[target] = [source]
        n_nodes = len(graph_)
        graph = dict()
        for node in range(n_nodes):
            graph[node]=graph_[node]

    # load features and labels
    if dataset_str == 'film':
        with open("{}/{}/out1_node_feature_label.txt".format(data_path, dataset_str), 'r') as f:
            data_str = f.readlines()
            title = data_str[0]
            body = data_str[1:]
            data_dim = int(title.split(':')[1].split(')')[0])
            n_nodes = len(body)
            features = np.zeros([n_nodes, data_dim])
            labels_ = np.zeros(n_nodes,dtype=int)
            for data in body:
                id_str, feat_str, label_str = data.split('\t')
                id = int(id_str)
                label = int(label_str)
                feat_str_list = feat_str.split(',')
                for feat_id in feat_str_list:
                    features[id][int(feat_id) - 1] = 1.
                labels_[id] = label
    else:
        with open("{}/{}/out1_node_feature_label.txt".format(data_path, dataset_str), 'r') as f:
            data_str = f.readlines()
            body = data_str[1:]
            example = body[0].split('\t')
            data_dim = len(example[1].split(','))
            n_nodes = len(body)
            features = np.zeros([n_nodes, data_dim])
            labels_ = np.zeros(n_nodes,dtype=int)
            for data in body:
                id_str, feat_str, label_str = data.split('\t')
                id = int(id_str)
                label = int(label_str)
                feat_str_list = feat_str.split(',')
                for i, feat in enumerate(feat_str_list):
                    features[id][i] = int(feat)
                labels_[id] = label
    labels = onehot_encoding(int(max(labels_)+1), labels_)
    n_per_label = [sum((labels_ == i).astype(int)) for i in class_ids]
    idx_train, idx_val, idx_test = [], [], []
    for id in range(len(class_ids)):
        ids_cl = (labels_ == class_ids[id]).astype(int).nonzero()[0].tolist()  # ids for examples belionging to current label
        split_tv, split_vt = int(n_per_label[id] * 0.6), int(n_per_label[id] * 0.8)  # index spliting train and val
        idx_train = idx_train + ids_cl[0:split_tv]
        idx_val = idx_val + ids_cl[split_tv:split_vt]
        idx_test = idx_test + ids_cl[split_vt:]


    num_class_total = labels.shape[1]
    class_labels = []
    for c in class_ids:
        label = sample_mask(c, labels.shape[1]) * 1
        class_labels.append(label)
    for k in graph:
        jm = (labels[k] == class_labels) # compare label with candidate classes
        jm = np.sum(jm, 1) # if label matches a candidate class exactly, num_class will be in jm after sum
        jm = (jm == num_class_total)
        if not np.any(jm):
            # if a node does not belong to current classes, then isolate it
            graph[k] = [k]
        else:
            to_pop = []
            for t in range(len(graph[k])):
                jm1 = (labels[graph[k][t]] == class_labels)  # compare label with candidate classes
                jm1 = np.sum(jm1, 1)  # if label matches a candidate class exactly, num_class will be in jm after sum
                jm1 = (jm1 == num_class_total)
                if not np.any(jm1):
                    # if a node connects to a neighbor not in current class, remove this neighbor
                    to_pop.append(graph[k][t])
            for p in to_pop:
                graph[k].remove(p)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # select only the involved classes in the labels
    y_train = y_train[:,class_ids]
    y_val = y_val[:, class_ids]
    y_test = y_test[:, class_ids]
    labels = labels[:, class_ids]

    # return multi-hop lists
    multi_nbs = multi_nb(graph, n_hop, flatten)
    return idx_train, idx_val, idx_test, graph, multi_nbs, features, y_train, y_val, y_test, labels

def load_ogb(dataset_str, class_ids, n_hop, flatten, directed = False, direction_reverse = False, args=None):
    # load edges and build the graph
    edge_pairs = np.load('./resources/datasets/{}/edge_feats.npy'.format(dataset_str))  # [2, 1166243]
    features = np.load('./resources/datasets/{}/node_feats.npy'.format(dataset_str))
    labels_ = np.load('./resources/datasets/{}/labels.npy'.format(dataset_str))
    n_edges = edge_pairs.shape[1]
    graph_ = dict()
    if not directed:
        for pair_id in range(n_edges):
            source = edge_pairs[0][pair_id]
            target = edge_pairs[1][pair_id]
            if source in graph_.keys():
                if target not in graph_[source]:
                    graph_[source].append(target)
            elif source not in graph_.keys():
                graph_[source] = [target]

            if target in graph_.keys():
                if source not in graph_[source]:
                    graph_[target].append(source)
            elif target not in graph_.keys():
                graph_[target] = [source]

    elif directed:
        if not direction_reverse:
            for pair_id in range(n_edges):
                source = edge_pairs[0][pair_id]
                target = edge_pairs[1][pair_id]
                if source in graph_.keys():
                    if target not in graph_[source]:
                        graph_[source].append(target)
                elif source not in graph_.keys():
                    graph_[source] = [target]
        elif direction_reverse:
            for pair_id in range(n_edges):
                source = edge_pairs[1][pair_id]
                target = edge_pairs[0][pair_id]
                if source in graph_.keys():
                    if target not in graph_[source]:
                        graph_[source].append(target)
                elif source not in graph_.keys():
                    graph_[source] = [target]

    graph = dict()
    n_nodes = labels_.shape[0]
    for node in range(n_nodes):
        if node not in graph_.keys():
            graph[node] = [node]
        else:
            graph[node] = graph_[node]

    # load features and labels
    labels = onehot_encoding(int(max(labels_)+1), labels_)
    n_per_label = [sum((labels_ == i).astype(int)) for i in class_ids]
    if args.standard_data_split:
        ogb_form_data = DglNodePropPredDataset(dataset_str.replace('_','-')) #, root='./data/ogb_downloaded') specificying root can manually change the download path of OGB data
        split_idx = ogb_form_data.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
            "test"].tolist()
    else:
        idx_train, idx_val, idx_test = [], [], []
        for id in range(len(class_ids)):
            ids_cl = (labels_ == class_ids[id]).astype(int).nonzero()[
                0].tolist()  # ids for examples belionging to current label
            split_tv, split_vt = int(n_per_label[id] * 0.6), int(n_per_label[id] * 0.8)  # index spliting train and val
            idx_train = idx_train + ids_cl[0:split_tv]
            idx_val = idx_val + ids_cl[split_tv:split_vt]
            idx_test = idx_test + ids_cl[split_vt:]

    num_class_total = labels.shape[1]
    class_labels = []
    for c in class_ids:
        label = sample_mask(c, labels.shape[1]) * 1
        class_labels.append(label)
    for k in graph:
        jm = (labels[k] == class_labels) # compare label with candidate classes
        jm = np.sum(jm, 1) # if label matches a candidate class exactly, num_class will be in jm after sum
        jm = (jm == num_class_total)
        if not np.any(jm):
            # if a node does not belong to current classes, then isolate it
            graph[k] = [k]
        else:
            to_pop = []
            for t in range(len(graph[k])):
                jm1 = (labels[graph[k][t]] == class_labels)  # compare label with candidate classes
                jm1 = np.sum(jm1, 1)  # if label matches a candidate class exactly, num_class will be in jm after sum
                jm1 = (jm1 == num_class_total)
                if not np.any(jm1):
                    # if a node connects to a neighbor not in current class, remove this neighbor
                    to_pop.append(graph[k][t])
            for p in to_pop:
                graph[k].remove(p)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # select only the involved classes in the labels
    y_train = y_train[:,class_ids]
    y_val = y_val[:, class_ids]
    y_test = y_test[:, class_ids]
    labels = labels[:, class_ids]

    # return multi-hop lists
    #multi_nbs = multi_nb_seq(graph, n_hop, flatten)
    multi_nbs = multi_nb(graph, n_hop, flatten)
    return idx_train, idx_val, idx_test, graph, multi_nbs, features, y_train, y_val, y_test, labels

def load_ogb_fast(dataset_str, class_ids, n_hop, flatten, directed = False, direction_reverse = False, args=None):
    # to fully utilize DGL to accelerate data loading,
    ogb_form_data = DglNodePropPredDataset(dataset_str.replace('_', '-')) #, root='./data/ogb_downloaded') specificying root can manually change the download path of OGB data
    edge_pairs = np.load('./resources/datasets/{}/edge_feats.npy'.format(dataset_str)) # [2, 1166243]
    features = np.load('./resources/datasets/{}/node_feats.npy'.format(dataset_str))
    labels_ = np.load('./resources/datasets/{}/labels.npy'.format(dataset_str))
    n_edges = edge_pairs.shape[1]
    graph_ = dict()
    ori_graph, _ = ogb_form_data[0]
    if not directed:
        ori_graph = dgl.add_reverse_edges(ori_graph)
    ori_graph = ori_graph.remove_self_loop() # avoid redundant self loop
    ori_graph = ori_graph.add_self_loop()

    # generate multi-hop neighborhood list
    multi_nbs=[]
    if flatten:
        for i in range(ori_graph.num_nodes()):
            to_be_find_nbs = [i]  # whose nbs are to be found, 1-hop nbs of center, 2-hop nbs of 1-hop, etc.
            nbs_current_node = []
            for h in range(n_hop):
                nbs = ori_graph.in_edges(to_be_find_nbs)[0].tolist()
                nbs_current_node.extend(nbs)
                to_be_find_nbs = nbs
            multi_nbs.append(nbs_current_node)
    else:
        for i in range(ori_graph.num_nodes()):
            to_be_find_nbs = [i]  # whose nbs are to be found, 1-hop nbs of center, 2-hop nbs of 1-hop, etc.
            nbs_current_node = []
            for h in range(n_hop):
                nbs = ori_graph.in_edges(to_be_find_nbs)[0].tolist()
                nbs_current_node.append(nbs)
                to_be_find_nbs = nbs
            multi_nbs.append(nbs_current_node)

    graph = dict()
    n_nodes = labels_.shape[0]
    for node in range(n_nodes):
        if node not in graph_.keys():
            graph[node] = [node]
        else:
            graph[node] = graph_[node]

    # load features and labels
    labels = onehot_encoding(int(max(labels_)+1), labels_)
    n_per_label = [sum((labels_ == i).astype(int)) for i in class_ids]
    if args.standard_data_split:
        ogb_form_data = DglNodePropPredDataset(dataset_str.replace('_','-')) #, root='./data/ogb_downloaded') specificying root can manually change the download path of OGB data
        split_idx = ogb_form_data.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"].tolist(), split_idx["valid"].tolist(), split_idx[
            "test"].tolist()
    else:
        idx_train, idx_val, idx_test = [], [], []
        for id in range(len(class_ids)):
            ids_cl = (labels_ == class_ids[id]).astype(int).nonzero()[
                0].tolist()  # ids for examples belionging to current label
            split_tv, split_vt = int(n_per_label[id] * 0.6), int(n_per_label[id] * 0.8)  # index spliting train and val
            idx_train = idx_train + ids_cl[0:split_tv]
            idx_val = idx_val + ids_cl[split_tv:split_vt]
            idx_test = idx_test + ids_cl[split_vt:]

    num_class_total = labels.shape[1]
    class_labels = []
    for c in class_ids:
        label = sample_mask(c, labels.shape[1]) * 1
        class_labels.append(label)
    for k in graph:
        jm = (labels[k] == class_labels) # compare label with candidate classes
        jm = np.sum(jm, 1) # if label matches a candidate class exactly, num_class will be in jm after sum
        jm = (jm == num_class_total)
        if not np.any(jm):
            # if a node does not belong to current classes, then isolate it
            graph[k] = [k]
        else:
            to_pop = []
            for t in range(len(graph[k])):
                jm1 = (labels[graph[k][t]] == class_labels)  # compare label with candidate classes
                jm1 = np.sum(jm1, 1)  # if label matches a candidate class exactly, num_class will be in jm after sum
                jm1 = (jm1 == num_class_total)
                if not np.any(jm1):
                    # if a node connects to a neighbor not in current class, remove this neighbor
                    to_pop.append(graph[k][t])
            for p in to_pop:
                graph[k].remove(p)


    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    # select only the involved classes in the labels
    y_train = y_train[:,class_ids]
    y_val = y_val[:, class_ids]
    y_test = y_test[:, class_ids]
    labels = labels[:, class_ids]

    # return multi-hop lists
    #multi_nbs = multi_nb_seq(graph, n_hop, flatten)
    #multi_nbs = multi_nb(graph, n_hop, flatten)
    return idx_train, idx_val, idx_test, graph, multi_nbs, features, y_train, y_val, y_test, labels

def load_data(data_path, dataset_str, class_ids, n_hop, flatten, args=None, standard = False):
    if dataset_str[0:4] == 'ogbn':
        if not standard:
            return load_ogb(dataset_str, class_ids, n_hop, flatten, args=args)
        else:
            return load_ogb_fast(dataset_str, class_ids, n_hop, flatten, args=args)
    elif dataset_str in ['cora','citeseer']:
        return load_data_part_G(data_path, dataset_str, class_ids, n_hop, flatten)
    else:
        return load_newdata_part_G(data_path, dataset_str, class_ids, n_hop, flatten)


class atten_classifier_GAT_o(nn.Module):
    # performance is around 74 on 7 classes, not very good
    def __init__(self, dim_original, dim_attention, num_class, inte = 'add'):
        super(atten_classifier_GAT_o, self).__init__()
        self.inte = inte
        self.w = nn.Linear(dim_original,dim_attention, bias=False)
        self.a = nn.Linear(2*dim_attention, 1, bias=False)
        if inte == 'add':
            self.classifier = nn.Linear(dim_attention, num_class)
        elif inte == 'concat':
            self.classifier = nn.Linear(2*dim_attention, num_class)

    def forward(self, atom, obj, cls = None):
        # atom[batch_size, lr+la, dim], obj[batch_size, dim]
        batch_size = obj.shape[0]
        dim_proto = obj.shape[-1]
        a1 = self.w(atom)
        o1 = self.w(obj).view(batch_size, 1, dim_proto)
        o2 = o1.repeat([1, a1.shape[1], 1])
        att_weights = self.a(torch.cat((a1, o2), dim=-1)).transpose(1,2) #[batch_size, 1, lr+la]
        message_a2o = att_weights.bmm(a1).squeeze() #[batch_size, dim]
        if self.inte == 'sum':
            preds_1 = self.classifier(o1.squeeze()+message_a2o)
        elif self.inte == 'concat':
            preds_1 = self.classifier(torch.cat((o1.squeeze(), message_a2o), dim=-1))
        return preds_1


def Pairwise_dis_shrink(x):
    # shrink distance between each pair of vectors in x
    cos_dis = torch.matmul(x, x.transpose(1,0))
    return torch.sum(cos_dis)

def Pairwise_dis_loss(x, d_min=0.1, mask_d = None, batch = None):
    # expect a set of tensor (2-d array), and return a loss to force pairwise distance larger than a threshold
    if batch:
        cos_dis = torch.triu(x.bmm(x.transpose(1,2)), diagonal=1) # get the upper triangle matrix without the diagonal elements
    else:
        cos_dis = torch.triu(x.mm(x.transpose(1,0)), diagonal=1) # get the upper triangle matrix without the diagonal elements
    if not mask_d is None:
        cos_dis = cos_dis*mask_d
    mask = (cos_dis<d_min).float()
    return -torch.sum(cos_dis * mask)

def Pairwise_dis_loss_(x, d_min=0.1):
    # expect a set of tensor (2-d array), and return a loss to force pairwise distance larger than a threshold
    cos_dis = torch.matmul(x, x.transpose(1,0))
    mask = (cos_dis>d_min).float()
    return torch.sum(cos_dis * mask)

class Relation_emb(nn.Module):
    # Given a vertex and its neighbors, embed each pair of vertices into an embedding
    def __init__(self, channel_list):
        super(Relation_emb, self).__init__()
        self.layer_list = nn.ModuleList([nn.Linear(in_channel, out_channel) for (in_channel, out_channel) in channel_list])

    #def forward(self, train_ids, train_features, adj_list, edge_feats=None):

class Component_prototypes(nn.Module):
    def __init__(self, dim_proto, dim_cls, n_proto_alloc, l_a, l_r, n_AFE, n_nbs_per_hop):
        super(Component_prototypes, self).__init__()
        self.l_a = l_a # number of attribute AFEs selected
        self.l_r = l_r
        self.n_nbs_per_hop = n_nbs_per_hop
        self.l_relu = nn.LeakyReLU()

        self.a_o_emb = nn.Linear((l_a+l_r)*dim_proto, dim_proto, bias=True) # embed concat atoms to objs
        self.o_c_emb = nn.Linear(dim_proto, dim_cls) # embed objs to be matched to cls
        #self.a_o_attention = nn.Linear((l_a+l_r)*dim_proto, (l_r+l_a)) # attention on atoms for objs
        self.a_o_attention = nn.Linear((l_a+l_r*sum(n_nbs_per_hop))*dim_proto, (l_r*sum(n_nbs_per_hop)+l_a)) # attention on atoms for objs
        self.a_o_att_w_GAT = nn.Linear(dim_proto, dim_proto, bias=False)
        self.a_o_att_a_GAT = nn.Linear(dim_proto*2, 1, bias=False)
        self.a_o_mask_emb = nn.Linear((l_a+l_r)*dim_proto, (l_a+l_r)*dim_proto)
        self.atoms = Parameter(torch.empty(n_proto_alloc[0], dim_proto).uniform_(0,0.5), requires_grad=True)  # atom prototypes
        self.objs = Parameter(torch.empty(n_proto_alloc[1], dim_proto).uniform_(0,0.5), requires_grad=True) # object prototypes
        self.cls = Parameter(torch.empty(n_proto_alloc[2], dim_cls).uniform_(0,0.5), requires_grad=True)  # class prototypes
        self.atom_stat = torch.zeros(n_proto_alloc[0], requires_grad=False)  # record number of embeds assigned to each atom
        self.obj_stat = torch.zeros(n_proto_alloc[1], requires_grad=False)  # record which nodes have been assigned to each object {k:set}
        self.cls_stat = torch.zeros(n_proto_alloc[2], requires_grad=False) # record which objs have been assigned to each class {k:set}
        self.obj_atom_map = torch.zeros([n_proto_alloc[1],n_proto_alloc[0]], requires_grad=False) # record to which atoms each object connects {obj:{atoms}}
        self.cls_obj_map = torch.zeros([n_proto_alloc[2],n_proto_alloc[1]], requires_grad=False) # record to which objs each cls connects

        #self.n_atom_total = n_proto_alloc
        self.num_atoms = 1
        self.num_atoms_old = 1
        self.atom_a_splits = [0] # record which AFE each attr atom corresponds to
        self.atom_r_splits = [0]  # record which embedding module each rela atom corresponds to
        self.AFE_attr_dict = [[0]]* n_AFE[0] # record which atoms belong to which AFE_attr
        self.AFE_rela_dict = [[0]] * n_AFE[1]  # record which atoms belong to which AFE_rela
        self.AFE_attr_alloc_rec = [0]* n_AFE[0] # record which AFE_attr has been allocated
        self.AFE_rela_alloc_rec = [0] * n_AFE[1]  # record which AFE_rela has been allocated
        self.num_objs = 1
        self.num_cls = 1

    def update(self, c_ids, embeddings, AFE_a_ids_selected, AFE_r_ids_selected, threshold, est_proto, threshold_c = 0.4): # nei_embs are embeddings of neighboring nodes
        # embeddings: [batch, n_AFE_a + n_AFE_r*n_nbs, d_proto]
        d_proto = embeddings.shape[-1]
        batch_size = len(c_ids)
        # allocate to GPU
        self.atom_stat, self.obj_stat, self.cls_stat = self.atom_stat.cuda(embeddings.get_device()), self.obj_stat.cuda(embeddings.get_device()), self.cls_stat.cuda(embeddings.get_device())

        # deal with current node (atoms)
        emb_set = [embeddings[:, i, :].contiguous().view(-1, d_proto) for i in range(len(AFE_a_ids_selected))]\
                  + [embeddings[:, i:i+sum(self.n_nbs_per_hop), :].contiguous().view(-1, d_proto) for i in range(len(AFE_a_ids_selected), embeddings.shape[1], sum(self.n_nbs_per_hop))]# seperate the embeddings from different AFEs
        atom_a_splits = self.atom_a_splits.copy()
        if atom_a_splits[-1] == self.num_atoms:
            self.num_atoms+=1
        atom_a_splits.append(self.num_atoms)

        soft_corres_atom_set = [emb_set[i].mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0)) for i in range(len(AFE_a_ids_selected) + len(AFE_r_ids_selected))] # compute sim between embs and aprotos

        if est_proto:
            # create new atoms for current nodes
            for id, embs in enumerate(emb_set[0:len(AFE_a_ids_selected)]):
                #AFE_id = AFE_a_ids_selected[id]
                soft_corres_atom = embs.mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0))
                corres_max = torch.max(soft_corres_atom, dim=1)[0]
                new_atoms = (corres_max < (1 - threshold))*(corres_max > 0)
                new_proto_indices = new_atoms.nonzero().squeeze(dim=-1)
                if len(new_proto_indices) != 0:
                    c_embs = embs
                    # avoid redundant atoms caused by similar embeddings
                    embs_for_new = c_embs[new_proto_indices].detach()
                    emb_sim = embs_for_new.mm(embs_for_new.transpose(1, 0))
                    emb_sim = torch.triu(emb_sim, diagonal=1)  # get upper tri-matrix
                    emb_sim = torch.max(emb_sim, 1)[0]  # get the max similarity between each emb and other embs
                    new_proto_indices = (emb_sim < (1 - threshold)).nonzero().squeeze(dim=-1)  #

                    with torch.no_grad():
                        self.atoms[self.num_atoms: self.num_atoms + len(new_proto_indices)] = embs_for_new[
                            new_proto_indices]
                        atoms_ = self.atoms.detach().cpu().numpy()
                    self.num_atoms += len(new_proto_indices)
                    self.num_atoms_old = self.num_atoms
                    atom_a_splits[-1] = self.num_atoms
                    soft_corres_atom_set[id] = c_embs.mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0))

            for id, embs in enumerate(emb_set[len(AFE_a_ids_selected):len(AFE_a_ids_selected)+len(AFE_r_ids_selected)]):
                id_real = id+len(AFE_a_ids_selected) # the beginning of id is len(AFE_r_ids_selected)
                #AFE_id = AFE_r_ids_selected[id]
                soft_corres_atom = embs.mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0))
                corres_max = torch.max(soft_corres_atom, dim=1)[0]
                new_atoms = (corres_max < (1 - threshold)) * (corres_max > 0)
                new_proto_indices = new_atoms.nonzero().squeeze(dim=-1)
                if len(new_proto_indices) != 0:
                    #c_embs = emb_set[id]
                    c_embs = emb_set[id_real]
                    # avoid redundant atoms caused by similar embeddings
                    embs_for_new = c_embs[new_proto_indices].detach()
                    emb_sim = embs_for_new.mm(embs_for_new.transpose(1, 0))
                    emb_sim = torch.triu(emb_sim, diagonal=1)  # get upper tri-matrix
                    emb_sim = torch.max(emb_sim, 1)[0]  # get the max similarity between each emb and other embs
                    new_proto_indices = (emb_sim < (1 - threshold)).nonzero().squeeze(dim=-1)  #

                    with torch.no_grad():
                        self.atoms[self.num_atoms: self.num_atoms + len(new_proto_indices)] = embs_for_new[new_proto_indices]
                        atoms_ = self.atoms.detach().cpu().numpy()
                    #self.AFE_rela_dict[AFE_id].extend(list(range(self.num_atoms, self.num_atoms+len(new_proto_indices))))
                    self.num_atoms += len(new_proto_indices)
                    self.num_atoms_old = self.num_atoms
                    atom_a_splits[-1] = self.num_atoms
                    soft_corres_atom_set[id_real] = c_embs.mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0))

            soft_corres_atom_set = [emb_set[i].mm(F.normalize(self.atoms[0:self.num_atoms], dim=-1).transpose(1, 0)) for
                                    i in range(len(AFE_a_ids_selected) + len(AFE_r_ids_selected))]
            max_logits_set = [soft_corres_atom.max(dim=1)[0] for soft_corres_atom in soft_corres_atom_set]
            max_logits_set = [max_logits.view(-1, 1) for max_logits in max_logits_set]
            n_AFE_a, n_AFE_r = len(AFE_a_ids_selected), len(AFE_r_ids_selected)
            hard_corres_atom_set = [(soft_corres_atom_set[i] == max_logits_set[i]).float().view(batch_size, self.num_atoms) for i in range(n_AFE_a)] \
                                   + [(soft_corres_atom_set[i + n_AFE_a] == max_logits_set[i + n_AFE_a]).float().view(-1, self.num_atoms) for i in range(n_AFE_r)]  # [batch_size, lr+la, n_atoms]
            associated_protos_set = [hard_corres_atom_set[i].mm(self.atoms[0:self.num_atoms]) for i in range(len(hard_corres_atom_set))]

            # objs
        else:
            max_logits_set = [soft_corres_atom.max(dim=1)[0] for soft_corres_atom in soft_corres_atom_set] # n_AEM_c * [batch*lr+la]
            max_logits_set = [max_logits.view(-1, 1) for max_logits in max_logits_set]
            #max_logits = torch.cat([m.view(batch_size, self.l_r+self.l_a) for m in max_logits_], dim=1) # [batch, n_AEM_c * (lr+la)]
            #voted_AEM = int(np.argmax(votes))
            sorted_ids_set = [torch.sort(max_logits.view(batch_size, -1), dim=1, descending=True)[1] for max_logits in max_logits_set]  # [batch_size, lr+la]
            #sorted_ids_n_set = [sorted_ids.detach().cpu().numpy() for sorted_ids in sorted_ids_set]
            #selected_sorted_ids_set = [sorted_ids[:, 0:(self.l_r + self.l_a)].view(batch_size, self.l_r + self.l_a, 1) for sorted_ids in sorted_ids_set]  # [batch_size, lr+la, 1] select fixed number of atoms

            #selected_sorted_ids_set = [sorted_ids.view(batch_size, 1, 1) for id,sorted_ids in enumerate(sorted_ids_set)]  # [batch_size, lr+la, 1] select fixed number of atoms

            '''
            votes = selected_sorted_ids//3
            vote_result = [(votes==i).sum() for i in range(n_AEM_c)]
            voted_AEM = int(np.argmax(vote_result))
            '''


            n_AFE_a, n_AFE_r = len(AFE_a_ids_selected), len(AFE_r_ids_selected)
            hard_corres_atom_set = [(soft_corres_atom_set[i] == max_logits_set[i]).float().view(batch_size, self.num_atoms) for i in range(n_AFE_a)]\
                + [(soft_corres_atom_set[i+n_AFE_a] == max_logits_set[i+n_AFE_a]).float().view(-1, self.num_atoms) for i in range(n_AFE_r)]# [batch_size, lr+la, n_atoms]
            '''
            hard_corres_atom_set = [(soft_corres_atom_set[i] == max_logits_set[i]).float().view(batch_size, len(self.AFE_attr_dict[AFE_a_ids_selected[i]])) for i in range(n_AFE_a)]\
                + [(soft_corres_atom_set[i+n_AFE_a] == max_logits_set[i+n_AFE_a]).float().view(batch_size, len(self.AFE_rela_dict[AFE_r_ids_selected[i]])) for i in range(n_AFE_r)]# [batch_size, lr+la, n_atoms]
            '''
            #hard_corres_atom_n = hard_corres_atom.detach().cpu().numpy()
            #associated_protos_set = [hard_corres_atom_set[i].mm(atom_set[i]) for i in range(len(hard_corres_atom_set))]
            associated_protos_set = [hard_corres_atom_set[i].mm(self.atoms[0:self.num_atoms]) for i in range(len(hard_corres_atom_set))]

        #associated_protos = hard_corres_atom.mm(self.atoms[0:self.num_atoms])
        associated_protos_set = [F.normalize(associated_protos, dim=-1).view(batch_size, -1, d_proto) for associated_protos in associated_protos_set] # also atoms
        associated_atoms = torch.cat(associated_protos_set, dim=1)

        # for objs

        # 1. use concat of atoms to generate attention weights
        a = associated_atoms.view(batch_size, -1) # [batch_size, n_AFE_a + n_AFE_r*n_nbs]
        atten_weight = self.a_o_attention(a).view(batch_size, 1, -1)# [batch_size, 1, n_AFE_a + n_AFE_r*n_nbs], attention weight for each atom
        #atten_weight = F.softmax(atten_weight, dim=-1) #has certain performance without this line
        b = associated_atoms.view(batch_size, -1, d_proto) # [batch_size, n_AFE_a + n_AFE_r*n_nbs, d_proto]
        obj_embs = F.normalize(atten_weight.bmm(b).view(batch_size, d_proto), dim=-1) # [batch, d_proto] each node gets an object level proto

        '''
        # 2. use current batch to generate attention weights
        a = associated_protos.view(batch_size, (self.l_r+self.l_a)*d_proto).mean(0)
        atten_weight = self.a_o_attention(a).view(1, (self.l_r+self.l_a), 1)
        atten_weight = F.softmax(atten_weight, 1)
        b = associated_protos.view(batch_size, self.l_r+self.l_a, d_proto)
        c = atten_weight * b
        obj_embs = c.mean(1)
        '''
        '''
        # 3. embed concat of atoms into objs
        a = associated_protos.view(batch_size, (self.l_r + self.l_a) * d_proto)
        obj_embs = self.a_o_emb(a) # [batch_size, d_proto]
        '''
        '''
        # 4. mask the concat of atoms for objs
        a = associated_protos.view(batch_size, (self.l_r + self.l_a) * d_proto)
        mask = self.a_o_mask_emb(a).sigmoid()
        obj_embs = mask*a
        
        # 5. try the mean of atoms as objs
        obj_embs = associated_protos.view(batch_size, (self.l_a+self.l_r), d_proto).mean(1)
        
        # 6. attention (GAT version attention) update each atom then aggregate as an obj
        a = associated_protos.view(batch_size, (self.l_r+self.l_a), d_proto)
        b = self.a_o_att_w_GAT(a).view(batch_size, (self.l_r+self.l_a), 1, d_proto) # [batch_size, lr+la, 1, d_proto]
        c = b.repeat([1, 1, (self.l_r+self.l_a), 1]) # [batch_size, lr+la, lr+la, d_proto]
        d = c.transpose(1,2)
        e = torch.cat([c,d], dim=-1)
        atten_weight = self.a_o_att_a_GAT(e).squeeze()# [batch_size, lr+la, lr+la]
        atten_weight = F.softmax(atten_weight, dim=-1)
        ass_protos_up = atten_weight.bmm(a) # [batch_size, lr+la, d_proto]
        obj_embs = ass_protos_up.view(batch_size * (self.l_a+self.l_r), d_proto)
        '''
        # 7. attention (Transformer version) update atom to get obj

        # deal with current node (objects)
        soft_corres_obj = obj_embs.mm(F.normalize(self.objs[0:self.num_objs], dim=-1).transpose(1, 0))  # cosine dist between embeddings and protos [num_embeddings*num_protos]
        corres_max_obj = torch.max(soft_corres_obj, dim=1)[0]  # get the max value of each row
        new_objs = (corres_max_obj < (1 - threshold)) * (corres_max_obj > 0)  # denote which embeddings need establishing new objs
        new_obj_indices = new_objs.nonzero().squeeze(dim=-1)  # indices of positions of embeddings for new objs

        if est_proto:
            # create new atoms for current nodes
            if len(new_obj_indices) != 0:
                # avoid redundant atoms caused by similar embeddings
                obj_embs_for_new = obj_embs[new_obj_indices].detach()
                obj_emb_sim = obj_embs_for_new.mm(obj_embs_for_new.transpose(1, 0))
                obj_emb_sim = torch.triu(obj_emb_sim, diagonal=1)  # get upper tri-matrix
                obj_emb_sim = torch.max(obj_emb_sim, 1)[0] # get the max similarity between each emb and other embs
                new_obj_indices = (obj_emb_sim < (1 - threshold)).nonzero().squeeze(dim=-1)  #

                with torch.no_grad():
                    self.objs[self.num_objs: self.num_objs + len(new_obj_indices)] = obj_embs_for_new[new_obj_indices]
                self.num_objs += len(new_obj_indices)
                self.num_objs_old = self.num_objs
                soft_corres_obj = obj_embs.mm(F.normalize(self.objs[0:self.num_objs], dim=-1).transpose(1, 0))

            max_logits = soft_corres_obj.max(dim=1)[0]
            max_logits = max_logits.view(batch_size, 1)
            hard_corres_obj = (soft_corres_obj == max_logits).float()
            # update stat of atoms
            self.obj_stat[0:self.num_objs] += torch.mean(hard_corres_obj, 0)

            # objs
        else:
            max_logits = soft_corres_obj.max(dim=1)[0]
            max_logits = max_logits.view(batch_size, 1)
            hard_corres_obj = (soft_corres_obj == max_logits).float()

        associated_objs = hard_corres_obj.mm(self.objs[0:self.num_objs])
        associated_objs = F.normalize(associated_objs, dim=-1)

        # deal with current node (cls)
        # cls_embs = self.o_c_emb(associated_objs)
        cls_embs = associated_objs
        soft_corres_cls = cls_embs.mm(F.normalize(self.cls[0:self.num_cls], dim=-1).transpose(1,
                                                                                              0))  # cosine dist between embeddings and protos [num_embs*num_protos]
        corres_max_cls = torch.max(soft_corres_cls, dim=1)[0]  # get the max value of each row
        new_cls = (corres_max_cls < (1 - threshold_c)) * (corres_max_obj > 0)  # denote which embeddings need establishing new objs
        new_cls_indices = new_cls.nonzero().squeeze(dim=-1)  # indices of positions of embeddings for new objs

        if est_proto:
            # create new objs for current nodes
            if len(new_cls_indices) != 0:
                # avoid redundant atoms caused by similar embeddings
                cls_embs_for_new = F.normalize(cls_embs[new_cls_indices], p=2, dim=-1).detach()
                cls_emb_sim = cls_embs_for_new.mm(cls_embs_for_new.transpose(1, 0))
                cls_emb_sim = torch.triu(cls_emb_sim, diagonal=1)  # get upper tri-matrix
                cls_emb_sim = torch.max(cls_emb_sim, 1)[0]  # get the max similarity between each emb and other embs
                new_cls_indices = (cls_emb_sim < (1 - threshold_c)).nonzero().squeeze(dim=-1)  #

                with torch.no_grad():
                    self.cls[self.num_cls: self.num_cls + len(new_cls_indices)] = cls_embs_for_new[new_cls_indices]
                self.num_cls += len(new_cls_indices)
                self.num_cls_old = self.num_cls
                soft_corres_cls = cls_embs.mm(F.normalize(self.cls[0:self.num_cls], dim=-1).transpose(1, 0))

            max_logits = soft_corres_cls.max(dim=1)[0]
            max_logits = max_logits.view(batch_size, 1)
            hard_corres_cls = (soft_corres_cls == max_logits).float()
            # update stat of objs
            self.cls_stat[0:self.num_cls] += torch.mean(hard_corres_cls, 0)

        else:
            max_logits = soft_corres_cls.max(dim=1)[0]
            max_logits = max_logits.view(batch_size, 1)
            hard_corres_cls = (soft_corres_cls == max_logits).float()

        associated_cls = hard_corres_cls.mm(self.cls[0:self.num_cls])
        associated_cls = F.normalize(associated_cls, dim=-1)

        return associated_atoms, associated_objs, associated_cls, hard_corres_atom_set, hard_corres_obj

    def AFE_select(self, emb_attr_try, emb_rela_try, n_AFE_a_select, n_AFE_r_select, cool_down=False):  # select AFEs for the current task
        d_proto_r = emb_rela_try[0].shape[-1]
        n_AFE_attr, n_AFE_rela = len(emb_attr_try), len(emb_rela_try)  # number of sets of embedding modules
        # deal with attr AFE
        atom_set_attr = [self.atoms[self.AFE_attr_dict[i]] for i in range(n_AFE_attr)]
        soft_corres_atom_set_attr = [emb_attr_try[i].mm(F.normalize(atom_set_attr[i], dim=-1).transpose(1, 0)) for i in range(n_AFE_attr)]
        corres_max_set_attr = [torch.max(soft_corres_atom_set_attr[i], dim=1)[0] for i in range(n_AFE_attr)]  # get the max value of each row
        votes_attr = [i.mean().item() for i in corres_max_set_attr]
        votes_attr.sort(reverse=True)
        # deal with rela AFE
        atom_set_rela = [self.atoms[self.AFE_rela_dict[i]] for i in range(n_AFE_rela)]
        #soft_corres_atom_set_rela = [emb_rela_try[i].mm(F.normalize(atom_set_rela[i], dim=-1).transpose(1, 0)) for i in range(n_AFE_rela)]
        soft_corres_atom_set_rela = [emb_rela_try[i].view(-1,d_proto_r).mm(F.normalize(atom_set_rela[i], dim=-1).transpose(1, 0)) for i in range(n_AFE_rela)]
        corres_max_set_rela = [torch.max(soft_corres_atom_set_rela[i], dim=1)[0] for i in
                               range(n_AFE_rela)]  # get the max value of each row
        votes_rela = [i.mean().item() for i in corres_max_set_rela]
        votes_rela.sort(reverse=True)

        return [votes_attr.index(votes_attr[i]) for i in range(n_AFE_a_select)], [votes_rela.index(votes_rela[i]) for i in range(n_AFE_r_select)]

    def cls_update(self, associated_atom_ids, threshold_c):
        associated_atom = torch.matmul(associated_atom_ids.view(1, -1), self.cur_atoms)
        new_obj = torch.mean(associated_atom, 0).view(-1,1)
        co_dis = self.cur_cls.mm(new_obj).view(1,-1) # obj-cls cosine distances
        j = max(co_dis.view(-1))
        print('j is', j)
        print('codis is', co_dis)
        if max(co_dis.view(-1)) > (1-threshold_c):
            # if the associated cls already exists
            co_map = F.gumbel_softmax(co_dis, tau=0.1, hard=True).squeeze(dim=0)
            cls_id = co_map.nonzero().squeeze().item()  # return id of the associated cls
            print('comap is', co_map)
            print('cls id',cls_id)
            self.class_stat[cls_id].add(self.num_objs)
        else:
            # if a new cls needs to be established
            self.class_stat[self.num_cls] = set([self.num_objs])
            self.cls_obj_map[self.num_cls, self.num_objs] = 1
            self.num_cls += 1

    def normalize(self, indices=None):
        if not indices: # if user doesn't assign which prototypes to be updated
            save1 = self.atoms
            self.cur_atoms = F.normalize(self.cur_atoms, p=2, dim=1) # normalize each prototype embedding into unit ball
            if torch.sum(torch.isinf(self.atoms)) > 0:
                print('detect inf in 17')
                exit()
            if torch.sum(torch.isnan(self.atoms)) > 0:
                print('detect nan in 17')
                torch.save(save1, './useless/save1.pt')
                torch.save(self.atoms, './useless/save2.pt')
                exit()

        else:
            self.atoms[indices] = F.normalize(self.atoms[indices], p=2, dim=1) # only normalize the selected prototypes for efficiency
            if torch.sum(torch.isinf(self.atoms[indices])) > 0:
                print('detect inf in 18')
                exit()
            if torch.sum(torch.isnan(self.atoms[indices])) > 0:
                print('detect nan in 18')
                exit()

