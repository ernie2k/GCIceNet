import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class NN(nn.Module):
    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(NN, self).__init__()

        n_hid = int(2*n_feat)

        self.nn1 = nn.Linear(n_feat, n_hid)
        self.nn2 = nn.Linear(n_hid, n_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.leaky_relu(self.nn1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.nn2(x)
        return F.log_softmax(x, dim=1)
