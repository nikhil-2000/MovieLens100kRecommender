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


class NormalNNMetrics:

    def __init__(self, dataloader, model_file, dataset):
        self.embedder = CalcEmbeddings(dataloader, model_file)
        self.dataset = dataset

        self.embeddings, self.metadata, self.e_dict = self.embedder.get_embeddings()
        self.metadata = pd.Series(self.metadata)
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
            new_user.movies_watched_df(all_items)
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
            return self.metadata.iloc[best_indexes].to_list()
        else:
            return self.metadata.iloc[best_indexes].to_list() + [0] * (search_size - len(best_indexes))

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


def test_model(model_file):
    train_reader = Datareader("ua.base", size=0, training_frac=1)
    test_reader = Datareader("ua.test", size=0, training_frac=1)
    all_reader = Datareader("u.data", size=0, training_frac=1)

    metrics = []
    datasets = []
    dataloaders = []

    for reader in [train_reader, test_reader, all_reader]:
        data = TestDataset(reader.ratings_df, reader.user_df, reader.items_df)
        loader = DataLoader(data, batch_size=64)
        metric = NormalNNMetrics(loader, model_file, data)
        metrics.append(metric)
        datasets.append(data)
        dataloaders.append(loader)

    # train = TestDataset(train_reader.ratings_df, train_reader.user_df, train_reader.items_df)
    # test = TestDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)
    # all = TestDataset(all_reader.ratings_df, all_reader.user_df, all_reader.items_df)
    #
    # train_loader = DataLoader(train, batch_size=64)
    # test_loader = DataLoader(test, batch_size=64)
    # all_loader = DataLoader(all, batch_size=64)
    #
    # train_metrics = NormalNNMetrics(train_loader, model_file, train)
    # test_metrics = NormalNNMetrics(test_loader, model_file, test)
    # all_metrics = NormalNNMetrics(all_loader, model_file, all)


    model_names = ["Train", "Test", "All"]

    params = zip(metrics, datasets, dataloaders, model_names)

    search_size = 30
    tests = 1000
    samples = 1000
    output = PrettyTable()
    output.field_names = ["Data", "Hitrate", "Mean Rank"]

    ranks = []
    for metric, data, loader, name in params:

        print("\nTesting " + name)
        # for i in trange(tests):
        for i in range(tests):
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

            without_positive = data.item_ids[~data.item_ids.isin(user_rating.movie_id.unique())]
            random_ids = np.random.choice(without_positive, samples).tolist()
            all_ids = random_ids + [positive_id]
            random.shuffle(all_ids)

            # Find n Closest
            top_n = metric.top_n_questions(anchor, search_size)
            ranking = metric.rank_questions(all_ids, anchor)

            set_prediction = set(top_n)
            if positive_id in set_prediction:
                metric.hits += 1

            rank = ranking.index(positive_id)

            metric.ranks.append(rank)

        hr = metric.hitrate(tests)
        mr = metric.mean_rank()
        output.add_row([name, hr, mr])
        ranks.append(mr)

    return output  , ranks

if __name__ == '__main__':
    model_file = "train_100_64_0.05_Mar29_10-43-53.pth"

    tables = []
    model_files = []
    table = PrettyTable()
    table.field_names = ["Model", "Train", "Test", "All"]

    for model_file in tqdm(os.listdir("WeightFiles")):
        t, ranks = test_model("WeightFiles/" + model_file)
        model_files.append(model_file)
        tables.append(t)
        table.add_row([model_file] + [str(r) for r in ranks])

    print(table)
