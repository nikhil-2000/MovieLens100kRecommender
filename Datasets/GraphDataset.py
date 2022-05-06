
import dgl
import torch
import networkx as nx
from dgl.data import DGLDataset
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from tqdm import tqdm

from Datasets.Training import TrainDataset
from datareader import Datareader
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite


from helper_funcs import categories

class GraphDataset(DGLDataset):


    def __init__(self, user_ids, item_ids, train_interactions, val_interactions, test_interactions, node_embedding = 100):


        self.user_ids = user_ids
        self.item_ids = item_ids
        self.train = train_interactions
        self.validation = val_interactions
        self.test = test_interactions

        self.user_to_idx = {u: i for i, u in enumerate(user_ids)}
        self.idx_to_user = {i: u for i, u in enumerate(user_ids)}

        self.item_to_idx = {u: i for i, u in enumerate(item_ids)}
        self.idx_to_item = {i: u for i, u in enumerate(item_ids)}


        self.user_count, self.item_count = len(user_ids), len(item_ids)
        torch.manual_seed(1)
        self.user_embeddings_layer = nn.Embedding(self.user_count, node_embedding)
        torch.manual_seed(1)
        self.item_embeddings_layer = nn.Embedding(self.item_count, node_embedding)

        user_nodes = torch.arange(self.user_count).long()
        item_nodes = torch.arange(self.item_count).long()

        self.user_embeddings = self.user_embeddings_layer(user_nodes)
        self.item_embeddings = self.item_embeddings_layer(item_nodes)

        super(GraphDataset, self).__init__("Movie Lens")

    def process(self):

        self.train_graph = self.buildGraph(self.train)
        self.validation_graph = self.buildGraph(self.validation)
        self.test_graph = self.buildGraph(self.test)



    def buildGraph(self, interactions):

        user_edge = []
        item_edge = []

        for i, row in interactions.iterrows():
            user = row.user_id.item()
            item = row.movie_id.item()
            user_edge.append(self.user_to_idx[user])
            item_edge.append(self.item_to_idx[item])

        graph = dgl.heterograph({
            ("user","rating","movie"): (user_edge, item_edge)
            ,("movie", "rated_by", "user"): (item_edge, user_edge)
        }, num_nodes_dict={"user": self.user_count, "movie": self.item_count})

        graph.nodes["user"].data["feat"] = self.user_embeddings
        graph.nodes["movie"].data["feat"] = self.item_embeddings

        return graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data = Datareader("ua.base",size = 1000, training_frac=0.7, val_frac=0.3)
    # user_ids = data.user_df.index.unique().tolist()
    # item_ids = data.items_df.index.unique().tolist()
    #
    #
    # g_data = GraphDataset(user_ids, item_ids, data.train, data.validation, data.test)
    #

    dataset = TrainDataset(data.ratings_df, data.user_df, data.items_df)
    user = dataset.sample_user()
    df = user.interactions.sample(10)
