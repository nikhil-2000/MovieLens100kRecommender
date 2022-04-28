import random

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from Datasets.DatasetBase import User
from Datasets.Training import TrainDataset
from helper_funcs import categories

class UnweightedRandomWalk:

    def __init__(self, dataset : TrainDataset, closest = 20):

        self.dataset = dataset
        self.users = self.create_users()
        self.vectors, self.id_to_idx = self.get_user_scores()
        self.dists = pairwise_distances(self.vectors.transpose())

        self.closest = closest

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
        # print(len(all_skills))
        # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
        print("\nCreating Users")
        for user in tqdm(self.dataset.users):
            u_ratings = user_ratings[user_ratings["user_id"] == user.id]
            new_user = UserURW(u_ratings)
            # new_user.movies_watched_df(all_movies)
            users.append(new_user)

        return users

    def get_user_scores(self):
        score_matrix = np.zeros((len(categories), len(self.users)))
        ids_to_idx = {}
        for idx, user in enumerate(self.users):
            ids_to_idx[user.id] = idx
            score_matrix[:, idx] = user.get_vector()

        return score_matrix, ids_to_idx

    def get_closest_users(self, user_id):
        if user_id not in self.id_to_idx:
            user_id = random.choice(self.users).id
            return self.get_closest_users(user_id)

        idx = self.id_to_idx[user_id]
        dists_from_users = self.dists[:, idx]
        dists_from_users = dists_from_users[dists_from_users > 0]
        sorted_indexes = np.argsort(dists_from_users)
        closest_10_percent = self.closest
        best_idxs = sorted_indexes[:closest_10_percent]
        closest_users_dists = dists_from_users[best_idxs]
        closest_users = [self.users[i] for i in best_idxs]
        return closest_users_dists, closest_users


class UserURW:

    def __init__(self, user_ratings: pd.DataFrame):
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

        # if sum(averages) == 0:
        #     averages = [0.5 for c in categories]

        self.scores = np.array(averages)
        self.interactions = user_ratings
        self.user_movies = self.interactions.movie_id.unique()
        # self.has_watched = pd.DataFrame(columns=["movie_id", "watched"])

    def get_vector(self):
        # print(self.scores.transpose().shape)
        return self.scores.transpose()

    def get_questions_by_rating(self, anchor, max_ret):
        rating = anchor.rating
        potential_movies = self.interactions.loc[abs(self.interactions.rating - rating) < 1]

        if len(potential_movies) > max_ret:
            return potential_movies.movie_id.sample(max_ret).to_list()
        else:
            return potential_movies.movie_id.to_list()

    def get_correct_ids(self, movies):
        df = self.has_watched.loc[movies]
        df = df.to_numpy().squeeze()
        return df