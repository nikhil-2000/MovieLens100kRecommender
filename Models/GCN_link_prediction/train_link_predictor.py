import itertools
from datetime import datetime

import dgl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.tensorboard import SummaryWriter

from Datasets.GraphDataset_Old import GCNDataset
from Datasets.GraphDataset import GraphDataset
from Datasets.Training import TrainDataset
from Models.GCN_link_prediction.LinkPredictor import GraphSAGE, DotPredictor
from Models.GCN_link_prediction.run_eval import test_model
from datareader import Datareader
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn


import dgl.function as fn



def compute_loss(pos_score, neg_score):
    #Pos score a vector with positive edge scores i.e scores of edges which exist
    #Neg score a vector with negative edge scores i.e scores of edges which don't exist
    scores = torch.cat([pos_score, neg_score])
    # First edges are labeled 1 as they are positive and
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    #Should push likely edges towards having a score of 1 and unlikely edges have a score of 0
    return F.binary_cross_entropy_with_logits(scores, labels)


def compute_auc(pos_score, neg_score):
    #Same as above
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    #Measures difference between labels and actual guesses
    return roc_auc_score(labels, scores)

def split_data(g):
    # Selects edges in (u,v) format and shuffles all

    u, v = g.edges(etype="rating")
    n_of_rating_edges = g.number_of_edges(etype="rating")
    eids = np.arange(n_of_rating_edges)
    eids = np.random.permutation(eids)

    # Select 10% of edges as test edges, rest are training
    test_size = int(len(eids) * 0.1)
    train_size = n_of_rating_edges - test_size

    # The positive test edges are the first 10% of (u,v) pairs
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    # The positive training edges are the remaining 90% (u,v) pairs
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Gets adjacency matrix for graph
    adj = g.adj(etype="rating").to_dense().numpy()
    # Inverts adjacency matrix, all edges that didn't exists, now do except self loops
    adj_neg = 1 - adj  # - np.eye(g.number_of_nodes())
    # Select (u,v) pairs of all edges that don't exist
    neg_u, neg_v = np.where(adj_neg != 0)

    # Samples an equivalent number of negative edges that exist in the graph
    neg_eids = np.random.choice(len(neg_u), g.number_of_edges(etype="rating"))
    # The negative test edges are the first 10% of (u,v) pairs
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    # The negative training edges are the remaining 90% (u,v) pairs
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

    # Remove the positive test edges, this is what we will test the model on and predict if they exist
    train_g = dgl.remove_edges(g, eids[:test_size], etype="rating")
    train_g = dgl.remove_edges(train_g, eids[:test_size], etype="rated_by")

    positive_train = (train_pos_u, train_pos_v)
    positive_test = (test_pos_u, test_pos_v)
    negative_train = (train_neg_u, train_neg_v)
    negative_test = (test_neg_u, test_neg_v)

    all_negs = (neg_u, neg_v)

    return train_g, positive_train, positive_test, negative_train, negative_test, all_negs

def pos_neg_graphs(graph):

    pos_u, pos_v = graph.edges(etype="rating")
    n_of_rating_edges = graph.number_of_edges(etype="rating")
    eids = np.arange(n_of_rating_edges)


    all_neg_u, all_neg_v = get_negatives(graph)

    # Samples an equivalent number of negative edges that exist in the graph
    neg_eids = np.random.choice(len(all_neg_u), graph.number_of_edges(etype="rating"))
    neg_u, neg_v = all_neg_u[neg_eids], all_neg_v[neg_eids]

    return (pos_u, pos_v), (neg_u,neg_v)

def get_negatives(graph):
    adj = graph.adj(etype="rating").to_dense().numpy()
    # Inverts adjacency matrix, all edges that didn't exists, now do except self loops
    adj_neg = 1 - adj  # - np.eye(g.number_of_nodes())
    # Select (u,v) pairs of all edges that don't exist
    all_neg_u, all_neg_v = np.where(adj_neg != 0)
    return all_neg_u, all_neg_v

def construct_graph(graph, edges):
    u,v = edges
    new_graph = dgl.heterograph({("user", "rating", "movie"): (u,v),
                                   ("movie", "rated_by", "user"): (v,u)},
                                  num_nodes_dict={"user": graph.number_of_nodes("user"),
                                                  "movie": graph.number_of_nodes("movie")})

    return new_graph

def learn(train_graphs, val_graphs, args):
    # train_graph, train_pos, train_neg = train_graphs
    # val_graph, val_pos, val_neg = val_graphs
    run_name =  "_".join([str(a) for a in args]) + datetime.now().strftime("%b%d_%H-%M-%S")
    outpath = "WeightFiles/" + "_".join([str(a) for a in args]) + datetime.now().strftime("%b%d_%H-%M-%S")
    numepochs, lr, hidden_feats, embedding_size = args

    train_graph , _, _ = train_graphs
    in_feat = train_graph.ndata['feat']["user"].shape[1]
    model = GraphSAGE(in_feat, hidden_feats, embedding_size)
    pred = DotPredictor()

    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), pred.parameters()), lr = lr)

    train_writer = SummaryWriter(log_dir="runs/" + run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

    model.train()
    for e in range(numepochs):
        # forward the node embeddings
        for phase in ["train", "validation"]:
            if phase == "train":
                model.train()
                g, pos, neg = train_graphs
                writer = train_writer
            else:
                model.eval()
                g, pos, neg = val_graphs
                writer = val_writer

            h = model(g, g.ndata['feat'])
            # Adds score to each edge which represents a probability of the edge existing in both graphs
            pos_score = pred(pos, h)
            neg_score = pred(neg, h)
            loss = compute_loss(pos_score, neg_score)

            # backward
            if phase == "train":
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                if e % 5 == 0:
                    print('In epoch {}, loss: {}'.format(e, loss))

            writer.add_scalar("Epoch_loss/", loss, e + 1)

    with torch.no_grad():
        h = model(train_graph, train_graph.ndata['feat'])

        pos_score = pred(val_graphs[1], h)
        neg_score = pred(val_graphs[2], h)
        print('AUC', compute_auc(pos_score, neg_score))

    torch.save({
        'epoch': e,
        'model_state_dict': model.state_dict(),
        'optimzier_state_dict': optimizer.state_dict(),
        'loss': loss,
        'in_feat':in_feat,
        'hidden_feat':hidden_feats,
        'emb_size':embedding_size
    }, outpath + '.pth')
    return outpath + '.pth'

if __name__ == '__main__':

    train_reader = Datareader("ua.base", size = 0, training_frac=1, val_frac= 0.1)
    test_reader = Datareader("ua.test", size = 0, training_frac= 1)
    all_reader = Datareader("u.data", size = 0, training_frac= 0.7, val_frac=0.1)

    # user_ids = train_reader.user_df.index.unique().tolist()
    # item_ids = train_reader.items_df.index.unique().tolist()
    # dataset = GraphDatasetNew(user_ids, item_ids, train_reader.train, train_reader.validation, test_reader.ratings_df, node_embedding = 100)

    user_ids = all_reader.user_df.index.unique().tolist()
    item_ids = all_reader.items_df.index.unique().tolist()
    dataset = GraphDataset(user_ids, item_ids, all_reader.train, all_reader.validation, all_reader.test,
                           node_embedding=100)

    positive_train, negative_train = pos_neg_graphs(dataset.train_graph)
    positive_validation, negative_validation = pos_neg_graphs(dataset.validation_graph)
    positive_test, negative_test = pos_neg_graphs(dataset.test_graph)

    train_pos_g = construct_graph(dataset.train_graph, positive_train)
    # Builds negative graph using interactions that don't exist
    train_neg_g = construct_graph(dataset.train_graph, negative_train)

    val_pos_g = construct_graph(dataset.validation_graph, positive_validation)
    # Builds negative graph using interactions that don't exist
    val_neg_g = construct_graph(dataset.validation_graph, negative_validation)

    # Same again
    # test_pos_g = construct_graph(train_graph, positive_test)
    # # Builds negative graph using interactions that don't exist
    # test_neg_g = construct_graph(train_graph, negative_test)

    model_file = learn([dataset.train_graph, train_pos_g, train_neg_g],
                       [dataset.validation_graph, val_pos_g, val_neg_g],
                       [200, 0.01, 40 ,30])

    neg_u, neg_v = get_negatives(dataset.train_graph)

    full_negative_graph = construct_graph(dataset.test_graph, (neg_u, neg_v))
    full_negative_graph.ndata["feat"] = dataset.test_graph.ndata["feat"]

    #Test Dataset With Val Data
    # val_test_interactions = pd.concat([train_reader.validation, test_reader.ratings_df])
    trainDataset = TrainDataset(train_reader.train, train_reader.user_df, train_reader.items_df)
    testDataset = TrainDataset(test_reader.ratings_df, test_reader.user_df, test_reader.items_df)

    results = test_model(dataset.train_graph, full_negative_graph ,model_file, dataset, trainDataset, testDataset)
    print(results)