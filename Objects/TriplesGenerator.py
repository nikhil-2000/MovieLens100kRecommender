from __future__ import absolute_import

import os
import sys

from tqdm import tqdm

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

import torch
import pandas as pd
from Objects.Data_reader import Dataset
import numpy as np

class TriplesGenerator:

    def __init__(self, filename, include_categories=True, include_metrics=True, size=0):

        self.dataset = Dataset(filename, size=size)
        self.ratings_df = self.dataset.ratings_df
        self.user_df = self.dataset.user_df

        if include_metrics:
            self.dataset.add_metrics()

        self.movies_df = self.dataset.items_df

        cols_to_norm = self.movies_df.columns[5::]
        self.movies_df[cols_to_norm] = self.movies_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        self.movies_df.replace({np.nan: 0}, inplace = True)
        # self.movies_df.fillna(0)
        # self.movies_df.occupation = 1
        self.include_metrics = include_metrics
        self.include_categories = include_categories

        self.memory = {}
        self.memory_good = {}
        self.memory_bad = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        movie_id = self.movies_df.index[index]
        current_movie = self.movies_df.loc[movie_id]
        users_who_watched_movie = self.ratings_df[self.ratings_df.movie_id == movie_id]

        if len(users_who_watched_movie) == 0:
            random_user_id = self.dataset.user_ids.sample(1).item()
        else:
            random_user_id = users_who_watched_movie.sample(1).user_id.item()

        random_user = self.dataset.user_df.loc[random_user_id].squeeze()

        positive, negative = self.get_pair(random_user_id, movie_id)
        # positive = self.items.loc[self.items["movie_id"] == positive_id]
        # negative = self.items.loc[self.items["movie_id"] == negative_id]

        samples = [current_movie, positive, negative]
        return [self.extract_values(s) for s in samples]

        # return current_movie, positive.squeeze(), negative.squeeze()

    def extract_values(self, s: pd.Series):

        # s = s.to_dict()
        # Extracts Categories + Metrics
        if self.include_metrics and self.include_categories:
            data = s.to_list()[5:]
        elif self.include_categories:
            # Extracts Categories
            data = s.to_list()[5:-5]
        else:
            # Extracts Metrics
            data = s.to_list()[-5:]

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])
        return torch.Tensor(data)

    def get_pair(self,user_id,anchor_id):
        if not user_id in self.memory:
            self.memory[user_id] = self.ratings_df.loc[self.ratings_df["user_id"] == user_id]
            self.memory_good[user_id] = self.memory[user_id].loc[self.memory[user_id]["rating"] >= 3]
            self.memory_bad[user_id] = self.memory[user_id].loc[self.memory[user_id]["rating"] < 3]

        good_ratings = self.memory_good[user_id]
        if len(good_ratings) == 0:
            positive_id = self.dataset.movie_ids.sample(1)
        else:
            positive_id = good_ratings.sample(1).movie_id.item()

        bad_ratings = self.memory_bad[user_id]
        if len(bad_ratings) == 0:
            negative_id = self.dataset.movie_ids.sample(1)
        else:
            negative_id = bad_ratings.sample(1).movie_id.item()

        positive = self.movies_df.loc[positive_id].squeeze()
        negative = self.movies_df.loc[negative_id].squeeze()
        return positive,negative


    def get_pair_2(self, user, anchor_id):
        user_id = user.get("user_id").item()
        user_ratings = self.ratings_df.loc[self.ratings_df["user_id"] == user_id]
        user_ratings = self.ratings_df.loc[self.ratings_df.movie_id != anchor_id]

        good_ratings = user_ratings.loc[user_ratings["rating"] > 2.5]
        bad_ratings = user_ratings.loc[user_ratings["rating"] < 2.5]

        good_filter = self.movies_df.index.isin(good_ratings.index)
        good_movies = self.movies_df[good_filter]
        good_movie_vectors = [(i, self.extract_values(row)) for i, row in good_movies.iterrows()]

        bad_filter = self.movies_df.index.isin(bad_ratings.index)
        bad_movies = self.movies_df[bad_filter]
        bad_movie_vectors = [(i, self.extract_values(row)) for i, row in bad_movies.iterrows()]

        anchor = self.movies_df.loc[anchor_id]
        anchor_vec = self.extract_values(anchor)

        if len(good_movie_vectors) == 0:
            positive = self.movies_df.sample()
        else:
            diffs = [(i, abs(v - anchor_vec)) for i, v in good_movie_vectors]
            distances = [(i, torch.norm(v)) for i, v in diffs]
            positive_idx, _ = min(distances, key=lambda t: t[1])
            positive = self.movies_df.iloc[positive_idx]

        if len(bad_movie_vectors) == 0:
            negative = self.movies_df.sample()
        else:
            diffs = [(i, abs(v - anchor_vec)) for i, v in bad_movie_vectors]
            distances = [(i, torch.norm(v)) for i, v in diffs]
            negative_idx, _ = min(distances, key=lambda t: t[1])
            negative = self.movies_df.iloc[negative_idx]

        return positive, negative
