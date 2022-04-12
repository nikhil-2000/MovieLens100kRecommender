import random

import pandas as pd
import numpy as np
from tqdm import trange
"""
tensors = pd.read_csv("runs/Embeddings_training_GCN12-Apr_0019/00000/default/tensors.tsv", sep='\t', header = None).to_numpy()
metadata = pd.read_csv("runs/Embeddings_training_GCN12-Apr_0019/00000/default/metadata.tsv", sep='\t', header=0)
print(tensors)


def top_n_questions(movie_id, search_size):
    # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
    anchor_idx = metadata.index[metadata.id == movie_id].item()
    anchor_embedding = tensors[anchor_idx]
    dists = np.linalg.norm(tensors - anchor_embedding, axis=1)
    sorted_indexes = np.argsort(dists)
    best_indexes = sorted_indexes[1:search_size + 1]

    if len(best_indexes) >= search_size:
        return metadata.iloc[best_indexes].id.to_list()
    else:
        return metadata.iloc[best_indexes].to_list() + [0] * (search_size - len(best_indexes))


def rank_questions(ids, movie_id):
    anchor_idx = metadata.index[metadata.id == movie_id].item()
    anchor_embedding = tensors[anchor_idx]
    anchor_idxs = metadata.index[metadata.id.isin(ids)]
    idx_to_id = {i : metadata.iloc[idx].id.item() for i, idx in enumerate(anchor_idxs)}
    embeddings_to_rank = [tensors[i] for i in anchor_idxs]
    embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

    dists = np.linalg.norm(embeddings_to_rank - anchor_embedding, axis=1)
    sorted_indexes = np.argsort(dists)

    return [idx_to_id[i] for i in sorted_indexes]

if __name__ == '__main__':
    ranks = []
    for i in trange(1000):
        movie_id = metadata.sample(1).id.item()
        search = 50
        random_ids = metadata.sample(999).id.to_list() + [movie_id]
        random.shuffle(random_ids)

        top_n = top_n_questions(movie_id, search)
        ranking = rank_questions(random_ids, movie_id)

        # set_prediction = set(top_n)
        # if any([pos in set_prediction for pos in user_rating.movie_id]):
        #     metric.hits += 1

        rank = ranking.index(movie_id)

        ranks.append(rank)

    print(np.mean(ranks))
    """


import os
import random

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from datareader import Datareader
import numpy as np
import pandas as pd
import random


class GraphMetrics:

    def __init__(self, dataset):
        self.embeddings = pd.read_csv("runs/Embeddings_training_GCN12-Apr_1232/00000/default/tensors.tsv", sep='\t',
                              header=None).to_numpy()
        self.metadata = pd.read_csv("runs/Embeddings_training_GCN12-Apr_1232/00000/default/metadata.tsv", sep='\t', header=0)
        self.dataset = dataset

        self.e_dict = {row.id: self.embeddings[i] for i, row in self.metadata.iterrows()}

        self.users = self.create_users()
        self.hits = 0
        self.ranks = []

    def sample_user(self):
        return random.choice(self.users)

    def create_users(self):
        ratings = self.dataset.interaction_df

        user_ratings = ratings.sort_values("user_id")
        all_users = user_ratings.user_id.unique()
        all_items = self.dataset.item_ids

        users = []
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        print("\nCreating Users")
        # for user in tqdm(all_users):
        for user in all_users:
            u_ratings = user_ratings[user_ratings["user_id"] == user]
            new_user = User(u_ratings)
            # new_user.movies_watched_df(all_items)
            users.append(new_user)

        return users

    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        anchor_id = anchor.movie_id.item()
        anchor_embedding = self.e_dict[anchor_id]
        dists = np.linalg.norm(self.embeddings - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        best_indexes = sorted_indexes[1:search_size + 1]

        if len(best_indexes) >= search_size:
            return self.metadata.iloc[best_indexes].id.to_list()
        else:
            return self.metadata.iloc[best_indexes].id.to_list() + [0] * (search_size - len(best_indexes))

    def rank_questions(self, ids, anchor):
        anchor_id = anchor.movie_id.item()
        anchor = self.e_dict[anchor_id]

        embeddings_to_rank = [self.e_dict[k] for k in ids]
        embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

        dists = np.linalg.norm(embeddings_to_rank - anchor, axis=1)
        sorted_indexes = np.argsort(dists)

        return np.array(ids)[sorted_indexes].tolist()

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)


class User:

    def __init__(self, user_ratings: pd.DataFrame):
        self.id = user_ratings.iloc[0]["user_id"]
        self.ratings = user_ratings
        self.user_movies = self.ratings.movie_id.unique()
        self.has_watched = pd.DataFrame(columns=["movie_id", "watched"])

    def movies_watched_df(self, all_movies):
        self.has_watched.movie_id = all_movies
        self.has_watched.watched = self.has_watched.movie_id.isin(self.user_movies).astype(int)
        self.has_watched.set_index("movie_id", inplace=True)


def test_model():
    train_reader = Datareader("ua.base", size=0, training_frac=1, val_frac=0.2)
    test_reader = Datareader("ua.test", size=0, training_frac=1)

    metrics = []
    datasets = []
    dataloaders = []

    # for df in [train_reader.train, train_reader.validation, test_reader.train]:
    data = TestDataset(train_reader.ratings_df,  train_reader.user_df, train_reader.items_df)
    loader = DataLoader(data, batch_size=64)
    metric = GraphMetrics(data)
    metrics.append(metric)
    datasets.append(data)
    dataloaders.append(loader)

    model_names = ["Train", "Test", "Val"]

    params = zip(metrics, datasets, dataloaders, model_names)

    search_size = 30
    tests = 10000
    samples = 1000
    output = PrettyTable()
    output.field_names = ["Data", "Hitrate", "Mean Rank"]

    ranks = []
    hitrates = []
    for metric, data, loader, name in params:

        print("\nTesting " + name)
        # for i in trange(tests):
        for i in trange(tests):
            # Pick Random User
            total_ratings = 0
            while total_ratings < 5:
                user = metric.sample_user()
                user_rating, total_ratings = user.ratings, len(user.ratings)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_ratings), 2)
            anchor = user_rating.iloc[a_idx]
            anchor_id = anchor.movie_id.item()

            positive = user_rating.iloc[p_idx]
            positive_id = positive.movie_id.item()

            without_positive = metric.metadata.id[~metric.metadata.id.isin(user_rating.movie_id.unique())]
            random_ids = np.random.choice(without_positive, samples).tolist()
            all_ids = random_ids + [positive_id]
            random.shuffle(all_ids)

            # Find n Closest
            top_n = metric.top_n_questions(anchor, search_size)
            ranking = metric.rank_questions(all_ids, anchor)

            set_prediction = set(top_n)
            if any([pos in set_prediction for pos in user_rating.movie_id]):
                metric.hits += 1

            rank = ranking.index(positive_id)

            metric.ranks.append(rank)

        hr = metric.hitrate(tests)
        mr = metric.mean_rank()
        output.add_row([name, hr, mr])
        ranks.append(mr)
        hitrates.append(hr)

    return output, ranks, hitrates

def testWeightsFolder():
    tables = []
    model_files = []
    rank_table = PrettyTable()
    rank_table.field_names = ["Model", "Train", "Val", "Test"]

    hr_table = PrettyTable()
    hr_table.field_names = ["Model", "Train", "Val", "Test"]

    for model_file in tqdm(os.listdir("WeightFiles")):
        t, ranks, hitrates = test_model("WeightFiles/" + model_file)
        model_files.append(model_file)
        rank_table.add_row([model_file] + [str(r) for r in ranks])
        hr_table.add_row([model_file] + [str(h) for h in hitrates])
        print(t)
        break
        # print(rank_table)
        # print(hr_table)

if __name__ == '__main__':

    tables = []
    model_files = []
    rank_table = PrettyTable()
    rank_table.field_names = ["Model", "Train", "Val", "Test"]

    hr_table = PrettyTable()
    hr_table.field_names = ["Model", "Train", "Val", "Test"]

    t, r , h = test_model()
    print(t)


