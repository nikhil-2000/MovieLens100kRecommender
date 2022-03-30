import random
import time

import numpy as np
from torch.utils.data import DataLoader

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.URW.URW import UnweightedRandomWalk
from datareader import Datareader

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


class URW_Metrics:

    def __init__(self, dataset, model):
        self.dataset = dataset

        self.model = model

        self.hits = 0
        self.ranks = []

    def __len__(self):
        return len(self.dataset.ratings_df)

    def sample_user(self):
        return random.choice(self.model.users)

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
        distances, users = self.model.get_closest_users(user_id)


        movies_to_choose = convert_distances(distances, search_size)
        movies_to_choose = movies_to_choose.astype(int)

        for i, user in enumerate(users):
            max_qs = movies_to_choose[i]
            predicted = user.get_questions_by_rating(anchor, max_qs)
            top_n.extend(predicted)
            movies_to_choose[i] -= len(predicted)

        remaining = sum(movies_to_choose)
        break_loop = 0
        while remaining > 0:
            id = self.dataset.item_ids.sample(1).item()
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
        distances, users = self.model.get_closest_users(user_id)


        # t_1 = time.time()

        weights = convert_distances(distances, 100) / 100

        # t_2 = time.time()

        # for problem in ids:
        #     user_counts = [u.attempted_q(problem) * weights[i] for i, u in enumerate(closest_users)]
        #     movie_id_to_count[problem] = sum(user_counts)
        all_correct = np.zeros((len(users), len(ids)))
        for i, user in enumerate(users):
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



def convert_distances(distances, search_size):
    x = 1 / distances
    x = x / x.sum(axis=0)
    x = search_size * x
    x = np.floor(x)
    leftover = search_size - sum(x)
    x[0] += leftover
    return x


if __name__ == '__main__':
    datareader = Datareader("ua.base", size=0)
    train = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
    test = TestDataset(datareader.test, datareader.user_df, datareader.items_df)

    train_loader = DataLoader(train)
    test_loader = DataLoader(test)

    urw = UnweightedRandomWalk(train)

    train_metrics = URW_Metrics(train,urw)
    test_metrics = URW_Metrics(test,urw)


    anchor = datareader.ratings_df.sample(1)
    random_ids = datareader.movie_ids.sample(100)
    t1 = train_metrics.top_n_questions(anchor, 100), train_metrics.rank_questions(random_ids, anchor)
    t2 = test_metrics.top_n_questions(anchor, 100), test_metrics.rank_questions(random_ids, anchor)


    print(t1)
    print(t2)
