import torch.nn as nn
from dgl import apply_each

import dgl.nn.pytorch as dglnn
import torch.nn.functional as F

class GCNModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_hidden = 40
        n_layers = 2


        self.n_layers = 2
        self.n_hidden = n_hidden
        embedding_size = 20

        etypes = ["rating","rated_by"]
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.HeteroGraphConv({
            "rating": dglnn.GraphConv(20,n_hidden),
            "rated_by": dglnn.GraphConv(24,n_hidden)
        }))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.HeteroGraphConv({
                etype: dglnn.GraphConv(n_hidden, n_hidden) for etype in etypes
            }))
        self.layers.append(dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(n_hidden, embedding_size) for etype in etypes
        }))
        # self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = apply_each(h,self.activation)
                # h = self.dropout(h)
        return h