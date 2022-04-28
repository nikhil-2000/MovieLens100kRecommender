import random

import torch

from helper_funcs import vector_features, add_metrics, categories
import pandas as pd


class DatasetBase:
    def __init__(self, interactions, users, items):
        # self.dataset = Dataset(filename, size=size)
        self.interaction_df = interactions
        self.user_df = users

        self.item_df = add_metrics(interactions, users, items)
        self.users, self.id_to_idx = self.create_users()
        self.item_ids = self.item_df.index.to_series()

    def __len__(self):
        return len(self.interaction_df)

    def sample_user(self):
        return random.choice(self.users)

    def create_users(self):
        ratings = self.interaction_df

        user_ratings = ratings.sort_values("user_id")
        all_users = user_ratings.user_id.unique()

        users = []
        id_to_idx = {}
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        print("\nCreating Users")
        # for user in tqdm(all_users):
        for i, user in enumerate(all_users):
            u_ratings = user_ratings[user_ratings["user_id"] == user]
            new_user = User(u_ratings)
            users.append(new_user)
            id_to_idx[new_user.id] = i

        return users, id_to_idx

    def reduce_users_films(self):
        u_ids = self.interaction_df.user_id.unique()
        i_ids = self.interaction_df.movie_id.unique()
        users = self.user_df.loc[self.user_df.index.isin(u_ids)]
        items = self.item_df.loc[self.item_df.index.isin(i_ids)]

        self.user_df = users
        self.item_df = items
        self.item_ids = self.item_df.index.to_series()

    def pick_cat(self, cats):
        is_one_idx = [i for i, c in enumerate(cats) if c == 1]
        return random.choice(is_one_idx)

class User:

    def __init__(self, user_ratings: pd.DataFrame):
        self.id = user_ratings.iloc[0]["user_id"]
        self.interactions = user_ratings
        self.user_movies = self.interactions.movie_id.unique()
        self.has_watched = pd.DataFrame(columns=["movie_id", "watched"])

        self.avg_rating = self.interactions.rating.mean()


