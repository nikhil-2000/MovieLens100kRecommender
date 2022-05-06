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
from Models.GCN_link_prediction.LinkPredictorMetrics import LinkPredictorMetrics
from datareader import Datareader
from helper_funcs import MRR, Recall, AveragePrecision


def get_edges_for_positives(anchor, positives, graph):

    user_id = anchor.user_id.item()
    user_id = user_id - 1
    pos_edges = positives.tolist() + [int(anchor.movie_id.item())]

    u, v = graph.edges(etype = "rating")
    eids = (u == user_id).nonzero()
    graph_ids = v[eids].squeeze().detach().tolist()
    found_edges = list(map(lambda x: x+1, graph_ids))

    same_length = len(pos_edges) == len(found_edges)
    same_items = set(positives) == set(pos_edges)

    return


def test_model(model_file, graphDataset, trainDataset, validationDataset,testDataset):

    """
    Take in a graph + Model

    """
    train_graph = graphDataset.train_graph
    train_metric = LinkPredictorMetrics(model_file, graphDataset, train_graph)
    val_metric = LinkPredictorMetrics(model_file, graphDataset, train_graph)
    test_metric = LinkPredictorMetrics(model_file, graphDataset, train_graph)
    # model_names = ["Train", "Val"]
    model_names = ["Train", "Val", "Test"]
    datasets = [trainDataset, validationDataset, testDataset]
    metrics = [train_metric, val_metric, test_metric]


    params = zip(metrics, datasets, model_names)


    results = []

    search_size = 100
    ap_length = 20
    tests = 500
    # samples = 1000

    for metric, dataset, name in params:
        print("\nTesting " + name)

        if name == "Train" or name == "Val":
            users = []
            while len(users) < tests:
                user = dataset.sample_user()
                total_interactions = len(user.interactions)
                if total_interactions > 5:
                    users.append(user)
        else:
            users = dataset.users

        for user in tqdm(users):
            # for i in range(tests):
            # Pick Random User
            # total_interactions = 0
            # while total_interactions < 5:
            #     user = dataset.sample_user()
            user_interactions, total_interactions = user.interactions, len(user.interactions)

            # Generate Anchor Positive
            a_idx, p_idx = random.sample(range(0, total_interactions), 2)
            anchor = user_interactions.iloc[a_idx]

            positive_ids = dataset.item_ids[dataset.item_ids.isin(user_interactions.movie_id.unique())]
            positive_ids = positive_ids[dataset.item_ids != anchor.movie_id.item()]


            if name == "Train":
                x = max(search_size, total_interactions - 1)
                top_n = metric.top_n_items(anchor, x)
                mrr = MRR(positive_ids, top_n[:search_size])
                # top_n = metric.top_n_items_edges(anchor, ap_length)
                ap = AveragePrecision(positive_ids, top_n[:ap_length])
                # top_n = metric.top_n_items_edges(anchor, total_interactions - 1)
                rec = Recall(positive_ids, top_n[:total_interactions - 1])
            else:
                x = max(search_size, total_interactions - 1)
                top_n = metric.top_n_items_edges(anchor, x)
                mrr = MRR(positive_ids, top_n[:search_size])
                # top_n = metric.top_n_items_edges(anchor, ap_length)
                ap = AveragePrecision(positive_ids, top_n[:ap_length])
                # top_n = metric.top_n_items_edges(anchor, total_interactions - 1)
                rec = Recall(positive_ids, top_n[:total_interactions - 1])


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



# def visualise(model_file, name):
#     datareader = Datareader("ua.base", size=1000, training_frac=1, val_frac=0.2)
#     add_embeddings_to_tensorboard(datareader, model_file,name)







if __name__ == '__main__':

    pass