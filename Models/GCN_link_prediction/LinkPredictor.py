
import dgl.nn.pytorch as dglnn
from dgl import apply_each
from torch import nn
import torch.nn.functional as F
import dgl.function as fn


class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, embedding_size):
        super(GraphSAGE, self).__init__()
        #Creates embeddings for size h_feat
        # self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        # self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv1 = dglnn.HeteroGraphConv({
            "rating": dglnn.SAGEConv(in_feats,h_feats, 'mean')
            ,"rated_by": dglnn.SAGEConv(in_feats,h_feats, 'mean')
        })

        self.conv2 = dglnn.HeteroGraphConv({
            "rating": dglnn.SAGEConv(h_feats,embedding_size, 'mean')
            ,"rated_by": dglnn.SAGEConv(h_feats,embedding_size, 'mean')
        })

        self.activation = F.relu

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = apply_each(h,self.activation)
        h = self.conv2(g, h)
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            #Gets embeddings for all nodes
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            # Apply edges will do this pairwise on all existing edges in the graph
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype = "rating")

            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][('user', 'rating', 'movie')][:, 0]
