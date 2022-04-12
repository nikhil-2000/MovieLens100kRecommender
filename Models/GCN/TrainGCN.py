import argparse
import datetime

import numpy as np
import torch
from dgl.dataloading import NeighborSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from Datasets.Training import TrainDataset
from Models.GCN.GCNModel import PinSAGEModel
from Models.GCN.GraphDataset import GCNDataset
from Models.GCN.sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator
from datareader import Datareader
from helper_funcs import categories
from pytorch_metric_learning import losses, miners

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")



def triplet_mining(args, model, train_loader, val_loader, train_writer, val_writer):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # For each batch of head-tail-negative triplets...
    steps = 0
    val_steps = 0


    tripletLoss = losses.TripletMarginLoss()
    miner = miners.TripletMarginMiner()

    for epoch_id in trange(args.num_epochs):
        model.train()
        epoch_losses = []

        for blocks in train_loader:
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)

            labels = blocks[-1].dstdata["label"]
            embeddings = model.get_repr(blocks)
            triplets = miner(embeddings, labels)

            loss = tripletLoss(embeddings, labels, triplets)
            optimizer.zero_grad()
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
            for blocks in val_loader:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                labels = blocks[-1].dstdata["label"]
                embeddings = model.get_repr(blocks)
                triplets = miner(embeddings, labels)

                loss = tripletLoss(embeddings, labels, triplets)
                loss = loss.detach().item()
                val_writer.add_scalar("triplet_loss/", loss, val_steps)
                val_steps += args.batch_size
                epoch_losses.append(loss)

        val_writer.add_scalar("Epoch_triplet_loss/", np.mean(epoch_losses), epoch_id + 1)

    return model

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
    train_ids = train_gcn_dataset.graph_movie_ids
    val_ids = val_gcn_dataset.graph_movie_ids


    #
    #
    #
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

    val_dataloader = DataLoader(torch.arange(val_graph.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)

    train_dataloader = DataLoader(
        torch.arange(train_graph.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)

    train_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_val")

    model = PinSAGEModel(train_graph, "movie", 128, 2)
    model = model.to(device)

    return triplet_mining(args, model, train_dataloader, val_dataloader, train_writer, val_writer)
"""
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # For each batch of head-tail-negative triplets...
    steps = 0
    val_steps = 0


    dataloader_it = iter(dataloader)
    val_loader = iter(val_dataloader)

    batches_per_epoch = (train_graph.number_of_edges() // args.batch_size) + 1
    val_batches_per_epoch = (val_graph.number_of_edges()// args.batch_size) + 1

    for epoch_id in range(args.num_epochs):
        model.train()
        epoch_losses = []
        for batch_id in trange(batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            optimizer.zero_grad()
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
            for batch_id in trange(val_batches_per_epoch):
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
"""


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cpu')  # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--run-name', type=str,
                        default="training_GCN" + datetime.datetime.now().strftime("%d-%b_%H%M"))
    args = parser.parse_args()
    datareader = Datareader("ua.base", size=000, training_frac=1, val_frac=0.3)
    m = main(args, datareader)

    all_dataset = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)

    evaluate(args,m, all_dataset)
