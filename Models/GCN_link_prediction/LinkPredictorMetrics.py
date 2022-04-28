import torch

from Datasets.GraphDataset_Old import GCNDataset
from Datasets.GraphDataset import GraphDataset
from Models.GCN_link_prediction.LinkPredictor import GraphSAGE
from Models.GCN_link_prediction.LinkPredictor import DotPredictor
from Models.MetricBase import MetricBase

import numpy as np

class LinkPredictorMetrics(MetricBase):
    

    def __init__(self, graph, model_file, graphDataset: GraphDataset):
        super(LinkPredictorMetrics, self).__init__()

        self.model_file = model_file
        self.graph = graph
        self.dataset = graphDataset
        self.pred = DotPredictor()

        self.user_edges, self.movie_edges = self.graph.edges(etype = "rating")
        self.user_edges = self.user_edges.detach().numpy().squeeze()
        self.movie_edges = self.movie_edges.detach().numpy().squeeze()


        self.set_model()
        self.add_embeddings_to_graph()
        self.get_score_edges()



    def set_model(self):
        checkpoint = torch.load(self.model_file)
        in_feats, hidden_feats, emb_size = checkpoint['in_feat'], checkpoint['hidden_feat'], checkpoint['emb_size']
        model = GraphSAGE(in_feats, hidden_feats, emb_size)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.model = model

    def add_embeddings_to_graph(self):
        h = self.model(self.graph, self.graph.ndata['feat'])
        self.graph.ndata["h"] = h

    def get_score_edges(self):

        self.scores = self.pred(self.graph,self.graph.ndata['feat']).squeeze().detach().numpy()


    def top_n_items(self, anchor, search_size):

        user_id = anchor.user_id.item()
        user_id =  self.dataset.user_to_idx[user_id]
        # idx = (self.graph.nodes("user")==user_id).nonzero()
        user_h = self.graph.ndata["h"]["user"][user_id].squeeze()
        # _id_dist_2 = []
        # for graph_movie_id in self.graph.nodes("movie"):
        #     movie_id = self.dataset.idx_to_item[graph_movie_id.item()]
        #     idx = (self.graph.nodes("movie") == graph_movie_id).nonzero()
        #     movie_h = self.graph.ndata['h']["movie"][idx].squeeze()
        #     score = torch.dot(user_h, movie_h)
        #     _id_dist_2.append((movie_id, score))
        #
        # _id_dist_2.sort( key= lambda x: x[1], reverse=True)
        # return [t[0] for t in _id_dist_2][:search_size]

        embeddings = self.graph.ndata["h"]["movie"]
        scores = torch.matmul(user_h.T, embeddings.T).tolist()
        items = self.graph.nodes("movie").numpy()

        sorted_idxs = np.argsort(scores)[::-1]
        graph_items = items[sorted_idxs[:search_size]]
        sorted_items = [self.dataset.idx_to_item[i] for i in graph_items]

        return sorted_items


    def top_n_items_edges(self, anchor, search_size):
        # user_id = anchor.user_id.item()
        # user_id = self.dataset.user_to_idx[user_id]
        #
        #
        #
        # scores = self.scores[user_eids]
        #
        # sorted_idxs = np.argsort(scores)[::-1]
        # sorted_items = items[sorted_idxs[:search_size]]
        #
        # sorted_items_2 = [self.dataset.idx_to_item[i] for i in sorted_items]
        #
        # sorted_items_1 =  [self.dataset.item_ids[item_idx] for item_idx in sorted_items]
        # assert sorted_items_1 == sorted_items_2
        user_id = anchor.user_id.item()
        user_id =  self.dataset.user_to_idx[user_id]
        # idx = (self.graph.nodes("user")==user_id).nonzero()
        user_h = self.graph.ndata["h"]["user"][user_id].squeeze()
        embeddings = self.graph.ndata["h"]["movie"]
        scores = torch.matmul(user_h.T, embeddings.T).tolist()
        items = self.graph.nodes("movie").numpy()
        user_eids = (self.user_edges == user_id).nonzero()

        sorted_idxs = np.argsort(scores)[::-1]
        graph_items = items[sorted_idxs]
        connected_to = self.movie_edges[user_eids]
        graph_items = [i for i in graph_items if i in connected_to]
        sorted_items = [self.dataset.idx_to_item[i] for i in graph_items]


        return sorted_items[:search_size]

