import math
import random

import pandas as pd

from tqdm import trange

from Metrics.CollabFilteringMetrics import CollabFilteringMetrics
from Metrics.NormallNNMetrics import NormalNNMetrics
from Metrics.RandomMetrics import RandomChoiceMetrics

from prettytable import PrettyTable
import numpy as np



def run_metrics(filename, model, samples = 1000, tests = 1000, size = 0):
    neural_network_metrics = NormalNNMetrics(filename, model)
    collab_filtering_metrics = CollabFilteringMetrics(filename, size= size)
    dataset = collab_filtering_metrics.dataset
    random_metrics = RandomChoiceMetrics(dataset)


    # models = [collab_filtering_metrics]
    models = [collab_filtering_metrics, neural_network_metrics, random_metrics]

    search_size = math.floor(len(dataset.movie_ids) / 100)

    same_count = 0

    # model_names = ["Random Walk", "Same Skill", "Random Choice", "Collab Filtering"]
    model_names = ["Collab Filtering", "Neural Network", "Random"]
    print("\nTests Begin")
    for i in trange(tests):
        # Pick Random User
        user = random.choice(collab_filtering_metrics.users)
        user_rating, total_ratings = user.ratings, len(user.ratings)
        # Generate Anchor Positive
        a_idx, p_idx = random.sample(range(0, total_ratings), 2)
        anchor = user_rating.iloc[a_idx]
        anchor_id = anchor.movie_id.item()

        positive = user_rating.iloc[p_idx]
        positive_id = positive.movie_id.item()

        without_positive = dataset.movie_ids[~dataset.movie_ids.isin(user_rating.movie_id.unique())]
        random_ids = np.random.choice(without_positive,samples).tolist()
        all_ids = random_ids + [positive_id]
        random.shuffle(all_ids)

        # Find n Closest
        for i,m in enumerate(models):
            top_n = m.top_n_questions(anchor, search_size)
            ranking = m.rank_questions(all_ids, anchor)

            set_prediction = set(top_n)
            if positive_id in set_prediction:
                m.hits += 1

            rank = ranking.index(positive_id)

            m.ranks.append(rank)

        # else:
        #     print(same_skill_metrics.ranks[-1])


    output = PrettyTable()
    output.field_names = ["Model", "Hitrate", "Mean Rank"]
    for i, name in enumerate(model_names):
        m = models[i]

        hr = m.hitrate(tests)
        mr = m.mean_rank()
        output.add_row([name, hr, mr])


    print(output)



if __name__ == '__main__':


    run_metrics("../ml-100k/ua.base", "../normal_nn_50_Mar23_19-22-16.pth", samples = 1000, tests = 5000, size = 0)
    run_metrics("../ml-100k/ua.test", "../normal_nn_50_Mar23_19-22-16.pth", samples = 1000, tests = 5000, size = 0)
    # run_metrics("../non_skill_builder_data_new.csv", "questions_dataset_Mar10_16-48-43.pth", samples = 500, tests = 1000, size = 0000)

