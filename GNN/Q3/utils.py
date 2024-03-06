import networkx as nx
import numpy as np
import torch
from torch_geometric.datasets import CoraFull, Planetoid

from normalization import fetch_normalization, row_normalize
# random seed setting
seed = 43
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_datasets(dataset_name):
    """ Get datasets CoraFull and CiteSeer from torch_geometric """
    if dataset_name.lower() == "corafull":
        dataset = CoraFull(root='data')
    else:
        dataset = Planetoid(root='data', name='CiteSeer')
    dataset = {"name": dataset.name, "num_classes": dataset.num_classes, "nodes":dataset.x, "edges": dataset.edge_index.T, "labels": dataset.y}
    return dataset

def to_numpy(dataset):
    for k, v in dataset.items():
        if type(v) == torch.Tensor:
            dataset[k] = v.numpy()
    return dataset

def split_data(dataset):
    classes_indexes = {i:[] for i in list(set(dataset["labels"].tolist()))}
    for index, label in enumerate(dataset["labels"]):
        classes_indexes[label.item()].append(index)
    train_indexes, val_indexes, test_indexes = [], [], []
    for indexes in classes_indexes.values():
        indexes = torch.tensor(indexes)[torch.randperm(len(indexes))]
        train_indexes.append(indexes[:int(len(indexes)*0.7)])
        val_indexes.append(indexes[int(len(indexes)*0.7):int(len(indexes)*0.8)])
        test_indexes.append(indexes[int(len(indexes)*0.8):])

    train_indexes, val_indexes, test_indexes = torch.cat(train_indexes), torch.cat(val_indexes), torch.cat(test_indexes)
    dataset["train_indexes"], dataset["val_indexes"], dataset["test_indexes"] = train_indexes, val_indexes, test_indexes
    dataset = to_numpy(dataset)
    return dataset

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features


def data_loader(dataset_name, normalization="AugNormAdj"):
    dataset = get_datasets(dataset_name)
    dataset = split_data(dataset)

    graph = {node:[] for node in range(len(dataset["nodes"]))}
    for edge in dataset["edges"]:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    degree = np.sum(adj, axis=1)
    features = dataset["nodes"]
    adj, features = preprocess_citation(adj, features, normalization)

    train_indexes, val_indexes, test_indexes = dataset["train_indexes"], dataset["val_indexes"], dataset["test_indexes"]
    labels = dataset["labels"]
    train_adj, train_features, learning_type = adj, features, "transductive"
    dataset = adj, train_adj, features, train_features, labels, train_indexes, val_indexes, test_indexes, degree, learning_type
    return dataset

