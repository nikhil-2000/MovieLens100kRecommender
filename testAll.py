import math
import random

from torch.utils.data import DataLoader
from tqdm import trange



from prettytable import PrettyTable
import numpy as np

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.NeuralNetwork.NormallNNMetrics import NormalNNMetrics
from Models.Random.RandomMetrics import RandomChoiceMetrics
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URWMetrics import URW_Metrics
from datareader import Datareader


def run_metrics(model, samples = 1000, tests = 1000, size = 0):
    datareader_train = Datareader("ua.base", size=0, training_frac=1)
    datareader_test = Datareader("ua.test", size=0, training_frac=1)

    train = TestDataset(datareader_train.ratings_df, datareader_train.user_df, datareader_train.items_df)
    test = TestDataset(datareader_test.ratings_df, datareader_test.user_df, datareader_test.items_df)

    train_loader = DataLoader(train)
    test_loader = DataLoader(test)

    datasets = {"train": train, "test":test}
    dataloaders = {"train": train_loader, "test":test_loader}
    phases = ["train", "test"]

    urw_dataset = TrainDataset(datareader_train.ratings_df, datareader_train.user_df, datareader_train.items_df)
    urw = UnweightedRandomWalk(urw_dataset)

    output = PrettyTable()
    output.field_names = ["Phase", "Model", "Hitrate", "Mean Rank"]

    for phase in phases:
        dataset, dataloader = datasets[phase] , dataloaders[phase]

        neural_network_metrics = NormalNNMetrics(dataloader, model, dataset)
        urw_metrics = URW_Metrics(dataset, urw)
        random_metrics = RandomChoiceMetrics(dataset)


    # models = [urw_metrics]
        metrics = [urw_metrics, neural_network_metrics, random_metrics]

        search_size = 20#math.floor(len(dataset.item_ids) / 100)


    # model_names = ["Random Walk", "Same Skill", "Random Choice", "Collab Filtering"]
        model_names = ["URW", "Neural Network", "Random"]
        print("\nTests Begin")
        for i in trange(tests):
            # Pick Random User
            total_ratings = 0
            while total_ratings < 3:
                user = neural_network_metrics.sample_user()
                user_rating, total_ratings = user.ratings, len(user.ratings)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_ratings), 2)
            anchor = user_rating.iloc[a_idx]
            anchor_id = anchor.movie_id.item()

            positive = user_rating.iloc[p_idx]
            positive_id = positive.movie_id.item()

            without_positive = dataset.item_ids[~dataset.item_ids.isin(user_rating.movie_id.unique())]
            random_ids = np.random.choice(without_positive,samples).tolist()
            all_ids = random_ids + [positive_id]
            random.shuffle(all_ids)

            # Find n Closest
            for i,m in enumerate(metrics):
                top_n = m.top_n_questions(anchor, search_size)
                ranking = m.rank_questions(all_ids, anchor)

                set_prediction = set(top_n)
                if any([pos in set_prediction for pos in user_rating.movie_id]):
                    m.hits += 1

                rank = ranking.index(positive_id)

                m.ranks.append(rank)

            # else:
            #     print(same_skill_metrics.ranks[-1])



        for i, name in enumerate(model_names):
            m = metrics[i]

            hr = m.hitrate(tests)
            mr = m.mean_rank()
            output.add_row([phase, name, hr, mr])


        print(output)



if __name__ == '__main__':

    model_file = "D:\My Docs/University\Year 4\Individual Project/MovieLens100kRecommender/64_40_0.1_0.5_ua.base.pth"
    run_metrics(model_file, samples = 1000, tests = 5000, size = 0)

    # run_metrics("../non_skill_builder_data_new.csv", "questions_dataset_Mar10_16-48-43.pth", samples = 500, tests = 1000, size = 0000)

