import os
import random
from datetime import datetime

import dgl
from dgl.dataloading import NeighborSampler
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from Datasets.GraphDataset_Old import GCNDataset
from Datasets.Training import TrainDataset
from Models.GCN.GCNMetrics import GCNMetrics
from datareader import Datareader
from helper_funcs import MRR, Recall, AveragePrecision


def test_model(model_file):
    train_reader = Datareader("ua.base", size=0, training_frac=1)
    test_reader = Datareader("ua.test", size=0, training_frac=1)

    metrics = []
    datasets = []
    dataloaders = []

    for reader in [train_reader, test_reader]:
        data = TrainDataset(reader.ratings_df, reader.user_df, reader.items_df)
        graphDataset = GCNDataset(data)

        train_nids = {"movie": graphDataset.graph_movie_ids}
        sampler = NeighborSampler([20])  # create a sampler
        loader = dgl.dataloading.DataLoader(
            graphDataset.graph,
            train_nids,
            sampler,
            batch_size=64,  # batch_size decides how many IDs are passed to sampler at once
            # other arguments
        )
        metric = GCNMetrics(loader, model_file, graphDataset)
        metrics.append(metric)
        datasets.append(data)
        dataloaders.append(loader)

    model_names = ["Train", "Val"]

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



def visualise(model_file, name):
    datareader = Datareader("ua.base", size=1000, training_frac=1, val_frac=0.2)
    add_embeddings_to_tensorboard(datareader, model_file,name)


def testWeightsFolder():

    train_table = PrettyTable()
    train_table.field_names = ["Model","Mean Reciprocal Rank","Average Precision","Recall By User"]
    validation_table = PrettyTable()
    validation_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    test_table = PrettyTable()
    test_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    for model_file in tqdm(os.listdir("WeightFiles")):
        model_file_path = "WeightFiles/" + model_file
        train_row, val_row = test_model(model_file_path)
        train_table.add_row([model_file] + [str(r) for r in train_row])
        validation_table.add_row([model_file] + [str(r) for r in val_row])
        # test_table.add_row([model_file] + [str(r) for r in test_row])
        print(train_table)
        print()
        print(validation_table)
        print()
        # print(test_table)
        with open("results.txt", "w") as f:
            f.write(str(train_table) + "\n" + str(validation_table) + "\n" + str(test_table))

        # visualise(model_file_path, "Embeddings")





if __name__ == '__main__':

    # train_table = PrettyTable()
    # train_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    # validation_table = PrettyTable()
    # validation_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    #
    # test_table = PrettyTable()
    # test_table.field_names = ["Model", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]
    # model_file = "128_20_0.5_0.01_Apr27_15-59-56.pth"
    # model_file_path = "WeightFiles/" + model_file
    # train_row, val_row = test_model(model_file_path)
    # train_table.add_row([model_file] + [str(r) for r in train_row])
    # validation_table.add_row([model_file] + [str(r) for r in val_row])
    # # test_table.add_row([model_file] + [str(r) for r in test_row])
    # print(train_table)
    # print()
    # print(validation_table)
    # print()
    testWeightsFolder()
    # print(test_table)
    # with open("results.txt", "w") as f:
    #     f.write(str(train_table) + "\n" + str(validation_table) + "\n" + str(test_table))

    # visualise(model_file_path, "Embeddings")