
import torch_geometric.transforms as T
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import SAGEConv, to_hetero
import torch
from torch_geometric.data import HeteroData
import dgl
from Datasets.Training import TrainDataset
from Models.GCN.GraphDataset import GCNDataset
from datareader import Datareader

# datareader = Datareader("ua.base", size=100)
#
# dataset = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
#
# dataset = GCNDataset(dataset)
# user_vectors = dataset.user_vectors()
# movie_vectors = dataset.movie_vectors()
# edge_idxs, edge_data = dataset.edge_rating_indexes()
# print()
# data = HeteroData()
# #
# data['movie'].x = movie_vectors # [num_papers, num_features_paper]
# data['user'].x = user_vectors # [num_authors, num_features_author]
#
# data['user', 'rates', 'movie'].edge_index = edge_idxs # [2, num_edges_cites]
#
# data['user', 'rates', 'movie'].edge_attr = edge_data  # [1, num_edges_cites]
#


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# print(data)
#
# model = GNN(hidden_channels=64, out_channels=300)
# model = to_hetero(model, data.metadata(), aggr='sum')


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['paper'].train_mask
    loss = F.cross_entropy(out['paper'][mask], data['paper'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

import dgl
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
ratings = dgl.heterograph(
    {('user', '+1', 'movie') : (np.array([0, 0, 1]), np.array([0, 1, 0])),
     ('user', '-1', 'movie') : (np.array([2]), np.array([1]))})

G = dgl.to_networkx(dgl.to_homogeneous(ratings))

nx.draw(G)
plt.show()

