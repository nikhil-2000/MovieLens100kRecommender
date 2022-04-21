import argparse
import datetime
from itertools import product

import numpy as np
import torch
from dgl.dataloading import NeighborSampler
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Datasets.Training import TrainDataset
from Models.GCN.GCNMetrics import test_model
from Models.GCN.GCNModel import PinSAGEModel
from Models.GCN.GraphDataset import GCNDataset
from Models.GCN.sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator
from datareader import Datareader
from helper_funcs import categories
from pytorch_metric_learning import losses, miners

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def main(args, datareader):
    # load and preprocess dataset
    train_dataset = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
    val_dataset = TrainDataset(datareader.validation, datareader.user_df, datareader.items_df)

    train_gcn_dataset = GCNDataset(train_dataset)
    val_gcn_dataset = GCNDataset(val_dataset)
    train_graph = train_gcn_dataset.graph
    val_graph = val_gcn_dataset.graph
    user_ntype = "user"
    item_ntype = "movie"

    batch_sampler = ItemToItemBatchSampler(
        train_graph, user_ntype, item_ntype, args.batch_size)

    val_batch_sampler = ItemToItemBatchSampler(
        val_graph, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(
        train_graph, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, train_graph, item_ntype, None)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)
    val_dataloader = DataLoader(val_batch_sampler,
        batch_size=args.batch_size,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    # val_dataloader = DataLoader(torch.arange(val_graph.number_of_nodes(item_ntype)),
    #     batch_size=args.batch_size,
    #     collate_fn=collator.collate_test,
    #     num_workers=args.num_workers)
    #
    # train_dataloader = DataLoader(
    #     torch.arange(train_graph.number_of_nodes(item_ntype)),
    #     batch_size=args.batch_size,
    #     collate_fn=collator.collate_test,
    #     num_workers=args.num_workers)

    train_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_val")

    model = PinSAGEModel(train_graph, "movie", args.embedding_size, 2)
    model = model.to(device)

    # return triplet_mining(args, model, train_dataloader, val_dataloader, train_writer, val_writer)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # For each batch of head-tail-negative triplets...
    steps = 0
    val_steps = 0


    dataloader_it = iter(dataloader)
    val_loader = iter(val_dataloader)

    batches_per_epoch = (train_graph.number_of_nodes("movie") // args.batch_size) + 1
    val_batches_per_epoch = (val_graph.number_of_nodes("movie")// args.batch_size) + 1

    for epoch_id in trange(args.num_epochs):
        model.train()
        epoch_losses = []
        for batch_id in range(batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            optimizer.zero_grad()
            loss = model(pos_graph, neg_graph, blocks).mean()
            loss.backward()
            optimizer.step()
            loss = loss.detach().item()
            train_writer.add_scalar("triplet_loss/", loss, steps)
            steps += args.batch_size
            epoch_losses.append(loss)

        train_writer.add_scalar("Epoch_triplet_loss/", np.mean(epoch_losses), epoch_id + 1)
        epoch_losses = []

        model.eval()
        with torch.no_grad():
            for batch_id in range(val_batches_per_epoch):
                pos_graph, neg_graph, blocks = next(val_loader)
                # Copy to GPU
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)

                loss = model(pos_graph, neg_graph, blocks).mean()
                loss = loss.detach().item()
                val_writer.add_scalar("triplet_loss/", loss, val_steps)
                val_steps += args.batch_size
                epoch_losses.append(loss)

        val_writer.add_scalar("Epoch_triplet_loss/", np.mean(epoch_losses), epoch_id + 1)

        # torch.save(model.state_dict(), args.run_name)

    return model



def evaluate(args, model, dataset):

    train_gcn_dataset = GCNDataset(dataset)
    train_graph = train_gcn_dataset.graph
    user_ntype = "user"
    item_ntype = "movie"

    neighbor_sampler = NeighborSampler(
        train_graph, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, train_graph, item_ntype, None)

    dataloader = DataLoader(
        torch.arange(train_graph.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)


    model.eval()
    with torch.no_grad():
        h_item_batches = []
        item_ids = []
        item_labels = []

        for blocks in dataloader:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)

            h_item_batches.append(model.get_repr(blocks))
            item_ids.append(blocks[-1].dstdata["_ID"])
            item_labels.append(blocks[-1].dstdata["label"])
        h_item = torch.cat(h_item_batches, 0)
        all_item_ids = torch.cat(item_ids, 0)
        all_item_labels = torch.cat(item_labels, 0)


    print(h_item.shape)
    movie_ids = train_gcn_dataset.graph_id_to_movie
    all_actual_ids = [movie_ids[i.detach().item()] for i in all_item_ids]
    all_categories = [categories[i.detach().item()] for i in all_item_labels]

    meta = list(zip(all_actual_ids, all_categories))
    writer = SummaryWriter("runs/Embeddings_" + args.run_name)
    writer.add_embedding(h_item, metadata=meta, metadata_header=["id", "categories"])
    writer.close()



if __name__ == '__main__':

    results = PrettyTable(["Params", "Hitrate", "Rank", "Pos Diff", "Neg Diff"])

    random_walk_lengths = [2,16]
    num_random_walks = [10,50]
    num_neighbors = [3,20]
    num_layers = [2]
    batch_size = [64, 256]
    lrs = [0.01,0.001]

    params = [random_walk_lengths, num_random_walks, num_neighbors, num_layers, batch_size, lrs]

    i = 0
    # for out in product(*params):
    for out in ([[2,50,20,2,64,0.001]]):
        random_walk, num_of_walks, neighbors, n_layers, batch_size, lr = out
        pars = "_".join(list(map(str, out)))
        print(pars)
        print(i)
        i += 1

        parser = argparse.ArgumentParser()
        parser.add_argument('--random-walk-length', type=int, default=random_walk)
        parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
        parser.add_argument('--num-random-walks', type=int, default=num_of_walks)
        parser.add_argument('--num-neighbors', type=int, default=neighbors)
        parser.add_argument('--num-layers', type=int, default=2)
        parser.add_argument('--embedding_size', type=int, default=50)
        parser.add_argument('--batch-size', type=int, default=64)
        parser.add_argument('--device', type=str, default='cpu')  # can also be "cuda:0"
        parser.add_argument('--num-epochs', type=int, default=20)
        parser.add_argument('--num-workers', type=int, default=0)
        parser.add_argument('--lr', type=float, default=lr)
        parser.add_argument('-k', type=int, default=10)
        parser.add_argument('--run-name', type=str,
                            default=pars + datetime.datetime.now().strftime("%d-%b_%H%M"))
        args = parser.parse_args()
        datareader = Datareader("ua.base", size=0, training_frac=1, val_frac=0.3)
        m = main(args, datareader)

        all_dataset = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)

        evaluate(args,m, all_dataset)

        summary_table, r , h , res = test_model("runs/Embeddings_" + args.run_name, datareader)
        results.add_row([pars] + res)
        print(results)
        with open("results.txt", "w") as f:
            f.write(str(results) + "\n")

