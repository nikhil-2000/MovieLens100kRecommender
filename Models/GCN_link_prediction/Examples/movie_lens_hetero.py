
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp
import dgl.nn.pytorch as dglnn


import dgl.data
from dgl import apply_each
from dgl.nn.pytorch import SAGEConv

from Datasets.GraphDataset_Old import GCNDataset
from Datasets.Training import TrainDataset
from datareader import Datareader

#Creates Homogenous graph
datareader = Datareader("ua.base", size = 0,training_frac=1)
train_dataset = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
dataset = GCNDataset(train_dataset)
g = dataset.graph

#Selects edges in (u,v) format and shuffles all
u, v = g.edges(etype = "rating")
n_of_rating_edges = g.number_of_edges(etype = "rating")
eids = np.arange(n_of_rating_edges)
eids = np.random.permutation(eids)

#Select 10% of edges as test edges, rest are training
test_size = int(len(eids) * 0.1)
train_size = n_of_rating_edges - test_size

# The positive test edges are the first 10% of (u,v) pairs
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
# The positive training edges are the remaining 90% (u,v) pairs
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Gets adjacency matrix for graph
adj = g.adj(etype="rating").to_dense().numpy()
# Inverts adjacency matrix, all edges that didn't exists, now do except self loops
adj_neg = 1 - adj #- np.eye(g.number_of_nodes())
#Select (u,v) pairs of all edges that don't exist
neg_u, neg_v = np.where(adj_neg != 0)

#Samples an equivalent number of negative edges that exist in the graph
neg_eids = np.random.choice(len(neg_u), g.number_of_edges(etype="rating"))
# The negative test edges are the first 10% of (u,v) pairs
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
# The negative training edges are the remaining 90% (u,v) pairs
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]


#Remove the positive test edges, this is what we will test the model on and predict if they exist
train_g = dgl.remove_edges(g, eids[:test_size], etype="rating")
train_g = dgl.remove_edges(train_g, eids[:test_size], etype="rated_by")




# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats, embedding_size):
        super(GraphSAGE, self).__init__()
        #Creates embeddings for size h_feat
        # self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        # self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        self.conv1 = dglnn.HeteroGraphConv({
            "rating": dglnn.SAGEConv((20,21),h_feats, 'mean')
            ,"rated_by": dglnn.SAGEConv((21,20),h_feats, 'mean')
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

#Builds positive graph using interactions that already exist and training data
# train_pos_g = dgl.graph((train_pos_u, train_pos_v), num_nodes=g.number_of_nodes())]
train_pos_g = dgl.heterograph({("user", "rating", "movie"): (train_pos_u, train_pos_v),
                               ("movie", "rated_by", "user"): (train_pos_v, train_pos_u)},
                              num_nodes_dict={"user": g.number_of_nodes("user"),
                                              "movie": g.number_of_nodes("movie")})
#Builds negative graph using interactions that don't exist
train_neg_g = dgl.heterograph({("user", "rating", "movie"): (train_neg_u, train_neg_v),
                               ("movie", "rated_by", "user"): (train_neg_v, train_neg_u)},
                              num_nodes_dict={"user": g.number_of_nodes("user"),
                                              "movie": g.number_of_nodes("movie")})

#Same again
test_pos_g = dgl.heterograph({("user", "rating", "movie"): (test_pos_u, test_pos_v),
                               ("movie", "rated_by", "user"): (test_pos_v, test_pos_u)},
                              num_nodes_dict={"user": g.number_of_nodes("user"),
                                              "movie": g.number_of_nodes("movie")})
#Builds negative graph using interactions that don't exist
test_neg_g = dgl.heterograph({("user", "rating", "movie"): (test_neg_u, test_neg_v),
                               ("movie", "rated_by", "user"): (test_neg_v, test_neg_u)},
                              num_nodes_dict={"user": g.number_of_nodes("user"),
                                              "movie": g.number_of_nodes("movie")})


import dgl.function as fn

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


model = GraphSAGE(train_g.ndata['feat']["user"].shape[1], 40,30)
# You can replace DotPredictor with MLPPredictor.
#pred = MLPPredictor(16)
pred = DotPredictor()

def compute_loss(pos_score, neg_score):
    #Pos score a vector with positive edge scores i.e scores of edges which exist
    #Neg score a vector with negative edge scores i.e scores of edges which don't exist
    scores = torch.cat([pos_score, neg_score])
    # First edges are labeled 1 as they are positive and
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    #Should push likely edges towards having a score of 1 and unlikely edges have a score of 0
    return F.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    #Same as above
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    #Measures difference between labels and actual guesses
    return roc_auc_score(labels, scores)


# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
model.train()
for e in range(200):
    # forward the node embeddings
    h = model(train_g, train_g.ndata['feat'])
    #Adds score to each edge which represents a probability of the edge existing in both graphs
    pos_score = pred(train_pos_g, h)
    neg_score = pred(train_neg_g, h)
    loss = compute_loss(pos_score, neg_score)

    # backward
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pos_score = pred(test_pos_g, h)
    neg_score = pred(test_neg_g, h)
    print('AUC', compute_auc(pos_score, neg_score))


full_negative_graph = dgl.heterograph({("user", "rating", "movie"): (neg_u, neg_v),
                               ("movie", "rated_by", "user"): (neg_v, neg_u)},
                              num_nodes_dict={"user": g.number_of_nodes("user"),
                                              "movie": g.number_of_nodes("movie")})


h = model(full_negative_graph, g.ndata["feat"])
g_with_scores = pred(full_negative_graph, h)
print()
