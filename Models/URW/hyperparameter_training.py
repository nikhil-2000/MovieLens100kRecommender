import random

from prettytable import PrettyTable
from tqdm import trange

from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URWMetrics import URW_Metrics
from datareader import Datareader
from helper_funcs import MRR, AveragePrecision, Recall


def test_model(train_reader, test_reader, urw = None):
    train_data = TrainDataset(train_reader.ratings_df, train_reader.user_df, train_reader.items_df)

    if urw == None:
        urw = UnweightedRandomWalk(train_data)

    metric = URW_Metrics(train_data, urw)

    metrics = []
    datasets = []
    metrics.append(metric)
    datasets.append(train_data)

    for reader in [train_reader, test_reader]:
        data = TestDataset(reader.ratings_df, reader.user_df, reader.items_df)
        metric = URW_Metrics(data, urw)
        metrics.append(metric)
        datasets.append(data)

    model_names = ["Train", "Test"]

    params = zip(metrics, datasets, model_names)

    results = []

    search_size = 100
    ap_length = 20
    tests = 1000
    # samples = 1000

    for metric, data, name in params:

        print("\nTesting " + name)
        for i in trange(tests):
            # for i in range(tests):
            # Pick Random User
            total_interactions = 0
            while total_interactions < 5:
                user = data.sample_user()
                user_interactions, total_interactions = user.interactions, len(user.interactions)
            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]

            positive_ids = data.item_ids[data.item_ids.isin(user_interactions.movie_id.unique())]
            positive_ids = positive_ids[data.item_ids != anchor.movie_id.item()]

            top_n = metric.top_n_items(anchor, search_size)
            mrr = MRR(positive_ids, top_n)
            top_n = metric.top_n_items(anchor, ap_length)
            ap = AveragePrecision(positive_ids, top_n)
            top_n = metric.top_n_items(anchor, total_interactions - 1)
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
    train_reader = Datareader("ua.base", size=000, training_frac=1)
    test_reader = Datareader("ua.test", training_frac=1)

    data = TrainDataset(train_reader.train, train_reader.user_df, train_reader.items_df)
    c = [5, 10, 20, 30, 40, 50, 75, 100]
    train_table = PrettyTable()
    train_table.field_names = ["k","Mean Reciprocal Rank","Average Precision","Recall By User"]

    test_table = PrettyTable()
    test_table.field_names = ["k", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    urw = UnweightedRandomWalk(data)

    for n in c:
        urw.closest = n
        train_row, test_row = test_model(train_reader, test_reader, urw)
        train_table.add_row([urw.closest] + [str(r) for r in train_row])
        test_table.add_row([urw.closest] + [str(r) for r in test_row])
        print(train_table)
        print()
        print(test_table)
        with open("results.txt", "w") as f:
            f.write(str(train_table) + "\n" + str(test_table) + "\n")