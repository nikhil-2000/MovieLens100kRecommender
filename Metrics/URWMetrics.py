import random
import time

import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from tqdm import tqdm
from Objects.Datasets.Data_reader import Dataset

"""
Create a user similarity matrix
Get the anchor and performance
Top N 
Get the K similar users, and find questions which they have a similar performance on
 - Prioritise similar skill items
Make a weighted choice of users, pick more questions from more similar users

Ranking Questions
Given 1000 ids
Get the K nearest users, rank by most completed questions
"""


class CollabFilteringMetrics:

    def __init__(self, filename, size=0, tests=0):
        self.dataset = Dataset(filename, size=size)

        self.users, self.skills = create_users(self.dataset.ratings_df, self.dataset.items_df)
        self.vectors, self.id_to_idx = get_user_scores(self.users, self.skills)
        self.dists = pairwise_distances(self.vectors.transpose())

        self.times = np.zeros((tests, 5))
        self.i = 0

        self.hits = 0
        self.ranks = []

    def __len__(self):
        return len(self.dataset.ratings_df)

    def sample_user(self):
        return random.choice(self.users)

    def hitrate(self, tests):
        return 100 * self.hits / tests

    def mean_rank(self):
        return sum(self.ranks) / len(self.ranks)

    def top_n_questions(self, anchor, search_size):
        # Get top 10% closest users
        # Assign each user a weight
        # Pick questions based on weight
        # Each question chosen should have the user performance be similair
        # i.e if anchor is correct, pick other correct questions from similar users

        top_n = []

        user_id = anchor.user_id.item()
        idx = self.id_to_idx[user_id]
        dists_from_users = self.dists[:, idx]
        dists_from_users = dists_from_users[dists_from_users > 0]
        sorted_indexes = np.argsort(dists_from_users)
        closest_10_percent = max(len(sorted_indexes) // 10, 10)
        best_idxs = sorted_indexes[:closest_10_percent]
        closest_users_dists = dists_from_users[best_idxs]
        closest_users = [self.users[i] for i in best_idxs]

        movies_to_choose = convert_distances(closest_users_dists, search_size)
        movies_to_choose = movies_to_choose.astype(int)

        for i, user in enumerate(closest_users):
            max_qs = movies_to_choose[i]
            predicted = user.get_questions_by_rating(anchor, max_qs)
            top_n.extend(predicted)
            movies_to_choose[i] -= len(predicted)

        remaining = sum(movies_to_choose)
        break_loop = 0
        while remaining > 0:
            id = self.dataset.movie_ids.sample(1).item()
            if not id in top_n:
                top_n.append(id)
                remaining -= 1
            else:
                break_loop += 1

            if break_loop > 50:
                top_n.extend([0] * remaining)
                remaining = -1

        return top_n

    def rank_questions(self, ids, anchor):
        start = time.time()
        user_id = anchor.user_id.item()
        idx = self.id_to_idx[user_id]
        dists_from_users = self.dists[:, idx]
        dists_from_users = dists_from_users[dists_from_users > 0]
        sorted_indexes = np.argsort(dists_from_users)
        closest_10_percent = max(len(sorted_indexes) // 10, 10)
        best_idxs = sorted_indexes[:closest_10_percent]
        closest_users_dists = dists_from_users[best_idxs]
        closest_users = [self.users[i] for i in best_idxs]
        # t_1 = time.time()

        weights = convert_distances(closest_users_dists, 100) / 100

        # t_2 = time.time()

        # for problem in ids:
        #     user_counts = [u.attempted_q(problem) * weights[i] for i, u in enumerate(closest_users)]
        #     movie_id_to_count[problem] = sum(user_counts)
        all_correct = np.zeros((len(closest_users), len(ids)))
        for i, user in enumerate(closest_users):
            user_performance = user.get_correct_ids(ids) * weights[i]
            all_correct[i] = user_performance
        # t_3 = time.time()

        all_correct = all_correct.sum(axis=0).tolist()
        # t_4 = time.time()

        counts = list(zip(ids, all_correct))
        highest = sorted(counts, key=lambda x: x[1], reverse=True)
        # t_5 = time.time()

        # ts = np.array([t_1 - start, t_2-t_1, t_3-t_2, t_4-t_3,t_5-t_4])
        # self.times[self.i] = ts
        # self.i += 1

        return [h[0] for h in highest]


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


        self.scores = np.array(averages)
        self.ratings = user_ratings
        self.user_movies = self.ratings.movie_id.unique()
        self.has_watched = pd.DataFrame(columns=["movie_id", "watched"])

    def movies_watched_df(self, all_movies):
        self.has_watched.movie_id = all_movies
        self.has_watched.watched = self.has_watched.movie_id.isin(self.user_movies).astype(int)
        self.has_watched.set_index("movie_id", inplace=True)

    def get_vector(self):
        # print(self.scores.transpose().shape)
        return self.scores.transpose()

    def get_questions_by_rating(self, anchor, max_ret):
        rating = anchor.rating
        potential_movies = self.ratings.loc[abs(self.ratings.rating - rating) < 1]

        if len(potential_movies) > max_ret:
            return potential_movies.movie_id.sample(max_ret).to_list()
        else:
            return potential_movies.movie_id.to_list()

    def get_correct_ids(self, movies):
        df = self.has_watched.loc[movies]
        df = df.to_numpy().squeeze()
        return df


def read_data(filename, size=0):
    dataset = pd.read_csv(filename, encoding="latin-1")
    if size > 0:
        dataset = dataset.head(size)

    dataset['skill_id'] = dataset['skill_id'].fillna(0)
    dataset["skill_id"] = dataset["skill_id"].apply(getSkillID)
    dataset["skill_id"] = dataset["skill_id"].apply(int)
    dataset = dataset[dataset["skill_id"] > 0]

    user_performance = dataset.groupby(["user_id", "skill_id"])["correct"].mean()
    user_performance = pd.DataFrame(user_performance)
    user_performance.reset_index(inplace=True)

    return dataset, user_performance


def getSkillID(ids):
    if isinstance(ids, int) or isinstance(ids, float):
        return ids

    ids = ids.split(",")
    return ids[0]


def create_users(ratings, movies):
    user_ratings = ratings.sort_values("user_id")

    categories = ['Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
                  'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    user_ratings[categories] = None
    all_users = user_ratings.user_id.unique()

    all_movies = movies.index.unique()

    print("\n Adding Categories")
    for id in tqdm(all_movies):
        movie_categories = movies.loc[id][categories]
        for c in categories:
            user_ratings.loc[user_ratings.movie_id == id, c] = movie_categories[c]



    users = []
    # print(len(all_skills))
    # skill_vectors = pd.DataFrame(columns=["user_id"] + list(all_skills))
    print("\nCreating Users")
    for user in tqdm(all_users):
        u_ratings = user_ratings[user_ratings["user_id"] == user]
        new_user = User(u_ratings, categories)
        new_user.movies_watched_df(all_movies)
        users.append(new_user)
    # user_ratings.reset_index(inplace = True)

    # users = list(filter(lambda x: x.has_skills(), users))

    return users, categories


def get_user_scores(users, categories):
    score_matrix = np.zeros((len(categories), len(users)))
    ids_to_idx = {}
    for idx, user in enumerate(users):
        ids_to_idx[user.id] = idx
        score_matrix[:, idx] = user.get_vector()

    return score_matrix, ids_to_idx


def convert_distances(distances, search_size):
    x = 1 / distances
    x = x / x.sum(axis=0)
    x = search_size * x
    x = np.floor(x)
    leftover = search_size - sum(x)
    x[0] += leftover
    return x


if __name__ == '__main__':
    CFM = CollabFilteringMetrics("../ml-100k/ua.base", size = 1000)
