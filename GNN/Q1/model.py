import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv


class MLP(nn.Module):
    def __init__(self, num_features, num_classes, num_layers, hidden_size, dropout=0.5, batch_norm=True):
        super(MLP, self).__init__()
        in_dim, out_dim = num_features, hidden_size
        self.mlp = nn.Sequential()
        i = 0
        for i in range(num_layers-1):
            self.mlp.add_module(f"linear{i}", nn.Linear(in_dim, out_dim))
            if batch_norm:
                self.mlp.add_module(f"dropout{i}", nn.BatchNorm1d(out_dim))
            self.mlp.add_module(f"activation{i}", nn.LeakyReLU())
            self.mlp.add_module(f"dropout{i}", nn.Dropout(dropout))
            in_dim, out_dim = hidden_size, hidden_size
        self.mlp.add_module(f"linear{i+1}", nn.Linear(in_dim, num_classes))
        self.mlp.add_module(f"activation{i+1}", nn.LogSoftmax(dim=1))

    def forward(self, data):
        predicted = self.mlp(data["nodes"])
        return predicted


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, batch_norm, dropout, softmax):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, in_dim)
        self.BN = nn.BatchNorm1d(in_dim) if batch_norm else lambda x:x
        self.dropout = nn.Dropout(dropout)
        self.activation1 = nn.LeakyReLU()
        self.gcn = GCNConv(in_dim, out_dim)
        self.activation2 = nn.LogSoftmax(dim=1) if softmax else lambda x:x 

    def forward(self, fea, edges):
        x = self.linear(fea)
        x = self.activation1(self.dropout(self.BN(x)))
        x = self.gcn(x, edges)
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

    def forward(self, data):
        x = data["nodes"]
        for i, gcn in enumerate(self.gcn_list[:-1]):
            fx = gcn(x, data["edges"].T)
            if self.skip_connection and i>0:
                x = fx + x
            else:
                x = fx
        x = self.gcn_list[-1](x, data["edges"].T)
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
    
    def forward(self, data):
        x = data["nodes"]
        for i, gcn in enumerate(self.gcn_list[:-1]):
            fx = gcn(x, data["edges"].T)
            fx = self.add_layer_label(fx, i)
            fx = self.proj_list[i](fx)
            if self.skip_connection:
                x = fx + x
            else:
                x = fx
        x = self.gcn_list[-1](x, data["edges"].T)
        return x
    

class GATLayer(nn.Module):
    def __init__(self, GAT_class, in_dim, out_dim, num_heads, batch_norm, dropout, softmax):
        super(GATLayer, self).__init__()
        self.linear = nn.Linear(in_dim, in_dim)
        self.BN = nn.BatchNorm1d(in_dim) if batch_norm else lambda x:x
        self.dropout = nn.Dropout(dropout)
        self.activation1 = nn.LeakyReLU()
        self.gat = GAT_class(in_dim, out_dim, head=num_heads)
        self.activation2 = nn.LogSoftmax(dim=1) if softmax else lambda x:x 

    def forward(self, fea, edges):
        x = self.linear(fea)
        x = self.activation1(self.dropout(self.BN(x)))
        x = self.gat(x, edges)
        x = self.activation2(x)
        return x

class GATModel(nn.Module):
    def __init__(self, GAT_class, num_features, num_classes, hidden_size, num_layers, num_heads, dropout=0.9, batch_norm=True):
        super(GATModel, self).__init__()
        in_dim, out_dim = num_features, hidden_size
        self.gat_list = nn.ModuleList()
        for _ in range(num_layers-1):
            gcn_layer = GATLayer(GAT_class, in_dim, out_dim, num_heads, batch_norm, dropout, softmax=False)
            self.gat_list.append(gcn_layer)
            in_dim, out_dim = hidden_size, hidden_size
        gcn_layer = GATLayer(GAT_class, in_dim, num_classes, num_heads, batch_norm, dropout=0, softmax=True)
        self.gat_list.append(gcn_layer)

    def forward(self, data):
        x = data["nodes"]
        for gat in self.gat_list[:-1]:
            x = gat(x, data["edges"].T)
        x = self.gat_list[-1](x, data["edges"].T)
        return x