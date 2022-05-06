from prettytable import PrettyTable
from torch.utils.data import DataLoader
import pickle

from Datasets.GraphDataset import GraphDataset
from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from Models.GCN_link_prediction.LinkPredictorMetrics import LinkPredictorMetrics
from Models.NeuralNetwork.NormallNNMetrics import NormalNNMetrics
from Models.URW.URW import UnweightedRandomWalk
from Models.URW.URWMetrics import URW_Metrics
from datareader import Datareader
from helper_funcs import AveragePrecision,Recall



def create_NN_metric(train_reader, test_reader):
    # testWeightsFolder(datareader)
    model_file = "1024_10_1_0.001_50000_May05_16-25-50_FINAL.pth"
    model_file_path = "NeuralNetwork/WeightFiles/" + model_file
    data = TestDataset(train_reader.train, train_reader.user_df, train_reader.items_df)
    loader = DataLoader(data, batch_size=1024)
    train_metric = NormalNNMetrics(loader,model_file_path,data)


    data = TestDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)
    loader = DataLoader(data, batch_size=1024)
    metric = NormalNNMetrics(loader, model_file_path, data, (train_metric.embeddings, train_metric.metadata))
    filename = "NN_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(metric, filehandler)

def create_GCN_metric(train_reader, test_reader):
    _in_embs = 50

    user_ids = train_reader.user_df.index.unique().tolist()
    item_ids = train_reader.items_df.index.unique().tolist()

    graphDataset = GraphDataset(user_ids, item_ids,
                           train_reader.train, train_reader.validation, test_reader.ratings_df,
                           node_embedding=_in_embs)

    model_file = "200_0.005_50_100_100May05_16-31-11.pth"
    # testDataset = TrainDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)
    model_path = "GCN_link_prediction/WeightFiles/"
    test_metric = LinkPredictorMetrics(model_path + model_file, graphDataset, graphDataset.train_graph)

    filename = "GCN_test_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(test_metric, filehandler)

def create_URW_metric(train_reader, test_reader):

    train_data = TrainDataset(train_reader.train, train_reader.user_df, train_reader.items_df)
    urw = UnweightedRandomWalk(train_data)

    urw.closest = 10

    test_data = TestDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)
    metric = URW_Metrics(test_data, urw)

    filename = "URW_test_metric.pickle"
    filehandler = open(filename, 'wb')
    pickle.dump(metric, filehandler)



def load_NN_metric():
    filename = "NN_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

def load_GCN_metric():
    filename = "GCN_test_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)


def load_URW_metric():

    filename = "URW_test_metric.pickle"
    filehandler = open(filename, 'rb')
    return pickle.load(filehandler)

def id_to_name(dataset, ids):
    movie_df = dataset.item_df
    return movie_df.loc[ids].movie_title.tolist()

if __name__ == '__main__':
    train_reader = Datareader("ua.base", size=0, training_frac=1, val_frac=0.25)
    test_reader = Datareader("ua.test", size=0, training_frac=1)
    # create_GCN_metric(train_reader, test_reader)
    # create_NN_metric(train_reader, test_reader)
    # create_URW_metric(train_reader, test_reader)

    neuralNetworkMetric : NormalNNMetrics = load_NN_metric()
    GCNMetric : LinkPredictorMetrics = load_GCN_metric()
    PESMetric : URW_Metrics = load_URW_metric()

    testDataset = PESMetric.dataset

    for i in range(10):
        total_interactions = 0
        while total_interactions < 10:
            user = testDataset.sample_user()
            total_interactions = len(user.interactions)

            search_size = 9
            anchor = user.interactions.iloc[0]
            actual_ids = user.interactions.movie_id.tolist()
            nn_top_n = neuralNetworkMetric.top_n_items(anchor, search_size)
            gcn_top_n = GCNMetric.top_n_items_edges(anchor, search_size)
            pes_top_n = PESMetric.top_n_items(anchor, search_size)

            # if Recall(actual_ids, gcn_top_n) < 0.3 or min(Recall(actual_ids, nn_top_n), Recall(actual_ids,pes_top_n)) < 0.1:
            #     total_interactions = 0

        print("GCN_precision:", AveragePrecision(actual_ids, gcn_top_n), Recall(actual_ids, gcn_top_n))
        print("NN_precision:", AveragePrecision(actual_ids, nn_top_n), Recall(actual_ids, nn_top_n))
        print("PES_precision:", AveragePrecision(actual_ids, pes_top_n), Recall(actual_ids, pes_top_n))

        results = [actual_ids, gcn_top_n,nn_top_n, pes_top_n]
        names = []

        for rs in results:
            names.append((id_to_name(testDataset, rs)))

        table = PrettyTable()
        table.field_names = ["Actual", "GCN", "NN", "PES"]
        names = list(zip(*names))
        for row in names:
            table.add_row(row)

        print(table)

    # Pick random user
    # Get top-n for them
    # Transfer IDs to name
    # Put it all in a single dataframe