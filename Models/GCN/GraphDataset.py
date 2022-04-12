import dgl
import torch
import networkx as nx
from dgl.data import DGLDataset
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from tqdm import tqdm

from Datasets.Training import TrainDataset
from datareader import Datareader
import numpy as np
import pandas as pd
from networkx.algorithms import bipartite


from helper_funcs import categories

"""
Create Graph with users and movies
Users have a vector of average movie rating per category
Movies have a category + metrics vector
Create an edge if user has rated the film
Edge weights are user ratings
"""

class GCNDataset(DGLDataset):

    def __init__(self, dataset):
        self.dataset = dataset
        self.dataset.reduce_users_films()
        super(GCNDataset, self).__init__(name = "MovieLens")
    def process(self):
        # G = nx.DiGraph()
        self.users = self.create_users()
        # user_dict, items_dict = self.build_dicts()
        # user_node_data = list(user_dict.items())
        # item_node_data = list(items_dict.items())
        # G.add_nodes_from(user_node_data, bipartite = 0)
        # G.add_nodes_from(item_node_data, bipartite = 1)


        user_ids = []
        movie_ids = []
        ratings = []

        u_to_graph_id = {user_id : i for i, user_id in enumerate(self.dataset.interaction_df.user_id.unique())}
        m_to_graph_id = {movie_id : i for i, movie_id in enumerate(self.dataset.interaction_df.movie_id.unique())}
        for i, row in self.dataset.interaction_df.iterrows():
            user = row.user_id.item()
            item = row.movie_id.item()
            rating = row.rating.item()
            ratings += [rating]
            user_ids.append(u_to_graph_id[user])
            movie_ids.append(m_to_graph_id[item])


        self.graph_user_ids = list(u_to_graph_id.values())
        self.graph_movie_ids = list(m_to_graph_id.values())
        self.graph_id_to_movie = {v : k for k,v in m_to_graph_id.items()}

        self.graph = dgl.heterograph({
            ("user","rating","movie"): (user_ids, movie_ids)
            ,("movie", "rated_by", "user"): (movie_ids, user_ids)
        })

        self.graph.nodes["user"].data["feat"] = self.user_vectors()
        self.graph.nodes["movie"].data["feat"], self.graph.nodes["movie"].data["label"] = self.movie_vectors()
        self.graph.edges[("user", "rating", "movie")].data["feat"] = torch.Tensor(ratings)
        self.graph.edges[("movie", "rated_by", "user")].data["feat"] = torch.Tensor(ratings)

        print(self.graph)


    def __getitem__(self, i):
        return self.graph

    def user_vectors(self):
        vs = [u.get_vector().transpose() for u in self.users]
        vs = np.array(vs).squeeze()
        return torch.Tensor(vs).float()


    def movie_vectors(self):
        vs = []
        labels = []
        for i, lbl in enumerate(self.dataset.item_ids):
            v , lbl = self.dataset[i]
            vs.append(v)
            labels.append(lbl)

        vs = torch.stack(vs)
        labels = torch.Tensor(labels)
        return vs.float(), labels.long()


    def __len__(self):
        return 1

    def create_users(self):
        ratings = self.dataset.interaction_df
        movies = self.dataset.item_df

        user_ratings = ratings.sort_values("user_id")


        user_ratings[categories] = None
        all_users = user_ratings.user_id.unique()

        all_movies = self.dataset.item_ids

        print("\n Adding Categories")
        for id in tqdm(all_movies):
            movie_categories = movies.loc[id][categories]
            for c in categories:
                user_ratings.loc[user_ratings.movie_id == id, c] = movie_categories[c]

        users = []
        print("\nCreating Users")
        for user in tqdm(all_users):
            u_ratings = user_ratings[user_ratings["user_id"] == user]
            new_user = User(u_ratings, categories)
            users.append(new_user)

        return users


    def build_dicts(self):
        user_dict = {}
        # scores = np.zeros((len(categories), len(self.users)))
        # for i, user in enumerate(self.users):
        #     scores[:, i] = user.get_vector()
        #
        # scaler = MinMaxScaler()
        # scaler.fit(scores)
        # scores = scaler.transform(scores)

        for i, user in enumerate(self.users):
            data = user.get_vector()
            if sum(data) > 0:
                user_dict["u_" + str(user.id)] = {"vector": data}

        item_dict = {}
        for i, id in enumerate(self.dataset.item_ids):
            data, _ = self.dataset[i]
            item_dict["i_"+ str(id)] = {"vector": data}

        return user_dict, item_dict

    def build_name_dicts(self):
        user_dict = {}

        for i, user in enumerate(self.users):
            data = user.get_vector()

            user_dict["u_" + str(user.id)] = {"vector": data}

        item_dict = {}
        for i, id in enumerate(self.dataset.item_ids):
            data, _ = self.dataset[i]
            item_dict["i_"+ str(id)] = {"vector": data}

        return user_dict, item_dict

class User:

    def __init__(self, user_ratings: pd.DataFrame, categories):
        self.id = user_ratings.iloc[0]["user_id"]
        category_count = {k: 0 for k in categories}
        ratings = {k: [] for k in categories}
        for i, row in user_ratings.iterrows():
            cs = row[categories][row == 1].index
            rating = row.rating
            for c in cs:
                category_count[c] += 1
                ratings[c].append(rating)

        averages = []
        for c in categories:
            if category_count[c] > 0:
                a = sum(ratings[c]) / category_count[c]
            else:
                a = 0

            averages.append(a)

        if sum(averages) == 0:
            averages = [0.5 for c in categories]

        self.scores = np.array(averages)
        self.ratings = user_ratings
        self.user_movies = self.ratings.movie_id.unique()
        self.has_watched = pd.DataFrame(columns=["movie_id", "watched"])


    def get_vector(self):
        # print(self.scores.transpose().shape)
        return self.scores.transpose()

def create_hetero_data(ratings, users, items):
    dataset = TrainDataset(ratings, users, items)

    dataset = GCNDataset(dataset)
    user_vectors = dataset.user_vectors()
    movie_vectors = dataset.movie_vectors()
    edge_idxs, edge_data = dataset.edge_rating_indexes()
    print()
    data = HeteroData()
    #
    data['movie'].x = movie_vectors  # [num_papers, num_features_paper]
    data['user'].x = user_vectors  # [num_authors, num_features_author]

    data['user', 'rates', 'movie'].edge_index = edge_idxs  # [2, num_edges_cites]

    data['user', 'rates', 'movie'].edge_attr = edge_data  # [1, num_edges_cites]

    data = T.ToUndirected()(data)
    return data


if __name__ == '__main__':
    datareader = Datareader("ua.base", size=1000)


    # dataset = create_hetero_data(datareader.ratings_df, datareader.user_df, datareader.items_df)
    train = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
    dataset = GCNDataset(train)

