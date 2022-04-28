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

"""
Create Graph with users and movies
Users have a vector of average movie rating per category
Movies have a category + metrics vector
Create an edge if user has rated the film
Edge weights are user ratings
"""

class GCNDataset(DGLDataset):

    def __init__(self, dataset : TrainDataset):
        self.interaction_df = dataset.interaction_df
        self.user_df = dataset.user_df
        self.item_df = dataset.item_df
        self.dataset = dataset
        super(GCNDataset, self).__init__(name = "MovieLens")
    def process(self):

        idx_to_type = {}
        idx_to_id = {}
        id_to_idx = {}

        for i, user_id in enumerate(self.interaction_df.user_id.unique()):
            idx_to_type[i] = "user"
            idx_to_id[i] = user_id
            id_to_idx[user_id] = i

        x = i
        for _, movie_id in enumerate(self.interaction_df.movie_id.unique()):
            idx_to_type[x] = "movie"
            idx_to_id[x] = movie_id
            id_to_idx[movie_id] = x
            x = x+1

        u, v = [],[]
        for i, row in self.interaction_df.iterrows():
            user = row.user_id.item()
            item = row.movie_id.item()
            u.append(id_to_idx[user])
            v.append(id_to_idx[item])


        self.graph = dgl.heterograph({
            ("_N","_E","_N"): (u, v)
            # ,("movie", "rated_by", "user"): (movie_ids, user_ids)
        })
        #
        # max_users = max(self.graph_user_ids) + 1
        # max_movie = max(self.graph_movie_ids) + 1

        # self.movie_embedding = nn.Embedding(max_movie, 20)
        #
        # #Pinsage Sampler doesn't look at user nodes, only the other nearby movie nodes, therefore the user feature vectors aren't needed
        # # self.graph.nodes["user"].data["features"] = self.user_vectors()
        # self.graph.nodes["movie"].data["features"], self.graph.nodes["movie"].data["label"]= self.movie_vectors()
        # self.graph.edges[("user", "rating", "movie")].data["features"] = torch.Tensor(ratings)
        # # self.graph.edges[("movie", "rated_by", "user")].data["features"] = torch.Tensor(ratings)
        # user_ids = torch.Tensor(self.graph_user_ids).long()
        # # movie_ids =  torch.Tensor(self.graph_movie_ids).long()
        self.embedding = nn.Embedding(self.graph.number_of_nodes(), 100)
        self.graph.ndata["feat"] = self.embedding(self.graph.nodes())
        print(self.graph)


    def __getitem__(self, i):
        return self.graph

    def user_vectors(self):
        vs = [torch.Tensor([u.avg_rating]) for u in self.dataset.users]
        vs = torch.stack(vs)
        return vs.float()


    def movie_vectors(self):
        vs = []
        labels = []
        for g_movie_id in self.graph_movie_ids:
            movie_id = self.graph_id_to_movie[g_movie_id]
            vector,lbl = self.dataset.get_movie_data(movie_id)
            vs.append(vector)
            labels.append(lbl)

        vs = torch.stack(vs)
        labels = torch.stack(labels)
        return vs.float(), labels


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


if __name__ == '__main__':
    datareader = Datareader("ua.base", size=1000)


    # dataset = create_hetero_data(datareader.ratings_df, datareader.user_df, datareader.items_df)
    train = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
    dataset = GCNDataset(train)

