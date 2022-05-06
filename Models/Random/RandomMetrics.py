import pandas as pd
import random

from prettytable import PrettyTable
from tqdm import trange, tqdm

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.MetricBase import MetricBase
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URWMetrics import URW_Metrics
from datareader import Datareader
from helper_funcs import MRR, Recall, AveragePrecision


class RandomChoiceMetrics(MetricBase):

    def __init__(self, dataset):

        self.items = dataset.item_ids
        # self.metadata = pd.DataFrame(self.metadata, columns=["problem_id", "skill_id", "skill_name"])
        super(RandomChoiceMetrics, self).__init__()


    def top_n_questions(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])

        if search_size <= len(self.items):
            return self.items.sample(search_size).to_list()
        else:
            metadata_size = len(self.items)
            return self.items.sample(metadata_size).to_list() + [0] * (search_size - metadata_size)

    def rank_questions(self, ids, anchor):
        random.shuffle(ids)
        return ids


def test_model():
    train_reader = Datareader("ua.base", size=000, training_frac=1)
    test_reader = Datareader("ua.test", size=000, training_frac=1)

    metrics = []
    datasets = []
    for reader in [train_reader, test_reader]:
        d = TestDataset(reader.ratings_df, reader.user_df, reader.items_df)
        metric = RandomChoiceMetrics(d)
        metrics.append(metric)
        datasets.append(d)


    model_names = ["Train", "Test"]

    params = zip(metrics, datasets, model_names)

    results = []

    search_size = 100
    ap_length = 20
    tests = 10000
    # samples = 1000

    for metric, data, name in params:

        print("\nTesting " + name)
        if name == "Train":
            users = []
            while len(users) < 500:
                user = data.sample_user()
                total_interactions = len(user.interactions)
                if total_interactions > 5:
                    users.append(user)
        else:
            users = data.users

        for user in tqdm(users):
            user_interactions, total_interactions = user.interactions, len(user.interactions)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]

            positive_ids = data.item_ids[data.item_ids.isin(user_interactions.movie_id.unique())]
            positive_ids = positive_ids[data.item_ids != anchor.movie_id.item()]

            top_n = metric.top_n_questions(anchor, search_size)
            mrr = MRR(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, ap_length)
            ap = AveragePrecision(positive_ids, top_n)
            top_n = metric.top_n_questions(anchor, total_interactions - 1)
            rec = Recall(positive_ids, top_n)

            metric.mrr_ranks.append(mrr)
            metric.average_precisions.append(ap)
            metric.recall.append(rec)

        # mr = metric.mean_rank()
        metric_mrr = metric.mean_reciprocal_rank()
        metric_ap = metric.get_average_precision()
        metric_rec = metric.get_recall()
        # ranks.append(mr)

        results.append([metric_mrr, metric_ap, metric_rec])

    return results


if __name__ == '__main__':

    train_table = PrettyTable()
    train_table.field_names = ["k","Mean Reciprocal Rank","Average Precision","Recall By User"]

    test_table = PrettyTable()
    test_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    train_row, test_row = test_model()
    train_table.add_row(["Train"] + [str(r) for r in train_row])
    test_table.add_row(["Test"] + [str(r) for r in test_row])
    print(train_table)
    print(test_table)
    with open("results.txt", "w") as f:
        f.write(str(train_table) + "\n" + str(test_table) + "\n")