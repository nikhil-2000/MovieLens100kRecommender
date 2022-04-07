import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
import torch_geometric
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from torch_geometric.data import RandomNodeSampler, DataLoader
from torch_geometric.loader import NeighborLoader, HGTLoader

from Models.GCN.GCNModel import get_model
from Models.GCN.GraphDataset import create_hetero_data
from datareader import Datareader
import networkx as nx
import matplotlib.pyplot as plt




def main(args):
    # load and preprocess dataset
    datareader = Datareader("ua.base", size=1000, training_frac=1, val_frac=0.3)
    train_data = create_hetero_data(datareader.train, datareader.user_df, datareader.items_df)
    val_data = create_hetero_data(datareader.validation, datareader.user_df, datareader.items_df)
    model = get_model(train_data)
    print(train_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loader = NeighborLoader(train_data, num_neighbors= {k : [10] for k in train_data.edge_types}, input_nodes="movie", batch_size = 64, shuffle = True)


    print(train_loader)
    print()

    # total_loss = 0
    # for i, batch in enumerate(train_loader):
    #     print(batch)
    #     optimizer.zero_grad()
    #     batch_size = batch['movie'].batch_size
    #     out = model(batch.x_dict, batch.edge_index_dict)
    #
    #     optimizer.step()

    # acc = evaluate(model, features, labels, val_mask)
    # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
    #       "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
    #                                      acc, n_edges / np.mean(dur) / 1000))

    print()
    # acc = evaluate(model, features, labels, test_mask)
    # print("Test accuracy {:.2%}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")

    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
