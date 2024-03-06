import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, dropout, softmax):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, in_dim)
        self.BN = nn.BatchNorm1d(in_dim) if batch_norm else lambda x:x
        self.dropout = nn.Dropout(dropout)
        self.activation1 = nn.LeakyReLU()
        self.gcn = GCNConv(in_dim, out_dim)
        self.activation2 = nn.LogSoftmax(dim=1) if softmax else lambda x:x 

    def forward(self, fea, edges, weights):
        x = self.linear(fea)
        x = self.activation1(self.dropout(self.BN(x)))
        x = self.gcn(x, edges, weights)
        x = self.activation2(x)
        return x

class GCNModel(nn.Module):
    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout=0.9, batch_norm=True, skip_connection=False):
        super(GCNModel, self).__init__()
        self.skip_connection = skip_connection
        in_dim, out_dim = num_features, hidden_size
        self.gcn_list = nn.ModuleList()
        for _ in range(num_layers-1):
            gcn_layer = GCNLayer(in_dim, out_dim, batch_norm, dropout, softmax=False)
            self.gcn_list.append(gcn_layer)
            in_dim, out_dim = hidden_size, hidden_size
        gcn_layer = GCNLayer(in_dim, num_classes, batch_norm, dropout=0, softmax=True)
        self.gcn_list.append(gcn_layer)

    def forward(self, x, adj):
        edges = adj._indices()
        weights = adj._values()
        for i, gcn in enumerate(self.gcn_list[:-1]):
            fx = gcn(x, edges, weights)
            if self.skip_connection and i>0:
                x = fx + x
            else:
                x = fx
        x = self.gcn_list[-1](x, edges, weights)
        return x

class NewGCNModel(nn.Module):
    """ New GCN model using new idea to resolve over-smoothing (Question3, last part) """
    def __init__(self, num_features, num_classes, hidden_size, num_layers, dropout=0.9, batch_norm=True, skip_connection=False):
        super(NewGCNModel, self).__init__()
        self.skip_connection = skip_connection
        in_dim, out_dim = num_features, hidden_size
        self.gcn_list = nn.ModuleList()
        self.proj_list = nn.ModuleList()
        for _ in range(num_layers-1):
            gcn_layer = GCNLayer(in_dim, out_dim, batch_norm, dropout, softmax=False)
            self.gcn_list.append(gcn_layer)
            self.proj_list.append(nn.Sequential(nn.Linear(out_dim+num_layers, out_dim), nn.Tanh()))
            in_dim, out_dim = hidden_size, hidden_size
        gcn_layer = GCNLayer(in_dim, num_classes, batch_norm, dropout=0, softmax=True)
        self.gcn_list.append(gcn_layer)

    def add_layer_label(self, x, layer_number):
        layer_label = torch.zeros(x.shape[0], len(self.gcn_list)).to(x.device)
        layer_label[:,layer_number] = 1
        x = torch.concat((x, layer_label), dim=1)
        return x
    
    def forward(self, x, adj):
        edges = adj._indices()
        weights = adj._values()
        for i, gcn in enumerate(self.gcn_list[:-1]):
            fx = gcn(x, edges, weights)
            fx = self.add_layer_label(fx, i)
            fx = self.proj_list[i](fx)
            if self.skip_connection:
                x = fx + x
            else:
                x = fx
        x = self.gcn_list[-1](x, edges, weights)
        return x