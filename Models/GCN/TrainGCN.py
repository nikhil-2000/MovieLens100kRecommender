import argparse
import datetime

import dgl
import torch
from dgl.dataloading import NeighborSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from Datasets.Training import TrainDataset
from Models.GCN.GCNModel import PinSAGEModel
from Models.GCN.GraphDataset import GCNDataset
from datareader import Datareader
from Models.GCN.sampler import ItemToItemBatchSampler, NeighborSampler, PinSAGECollator
import torchtext

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # load and preprocess dataset
    datareader = Datareader("ua.base", size=1000, training_frac=1, val_frac=0.3)
    train_dataset = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
    val_dataset = TrainDataset(datareader.validation, datareader.user_df, datareader.items_df)

    train_gcn_dataset = GCNDataset(train_dataset)
    g = train_gcn_dataset.graph
    user_ntype = "user"
    item_ntype = "movie"

    item_texts = {'title': train_gcn_dataset.dataset.item_df.movie_title.values}

    # Prepare torchtext dataset and vocabulary
    fields = {}
    examples = []
    for key, texts in item_texts.items():
        fields[key] = torchtext.legacy.data.Field(include_lengths=True, lower=True, batch_first=True)
    for i in range(g.number_of_nodes(item_ntype)):
        example = torchtext.legacy.data.Example.fromlist(
            [item_texts[key][i] for key in item_texts.keys()],
            [(key, fields[key]) for key in item_texts.keys()])
        examples.append(example)
    textset = torchtext.legacy.data.Dataset(examples, fields)
    for key, field in fields.items():
        field.build_vocab(getattr(textset, key))
        #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')





    val_gcn_dataset = GCNDataset(val_dataset)
    batch_sampler = ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    neighbor_sampler = NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    collator = PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train)

    dataloader_val = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test)

    dataloader_it = iter(dataloader)

    train_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + args.run_name + "_val")

    # Model
    model = PinSAGEModel(g, item_ntype, textset, args.hidden_dims, args.num_layers).to(device)
    # Optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # For each batch of head-tail-negative triplets...
    steps = 0

    for epoch_id in range(args.num_epochs):
        model.train()
        for batch_id in trange(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            # Copy to GPU
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = model(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_writer.add_scalar("triplet_loss/train", loss, steps)

            # model.eval()
            #
            # val_writer.add_scalar("triplet_loss/val", loss, steps)
            steps += args.batch_size
        # # Evaluate
        model.eval()
        with torch.no_grad():
            item_batches = torch.arange(g.number_of_nodes(item_ntype)).split(args.batch_size)
            h_item_batches = []
            for blocks in dataloader_val:
                for i in range(len(blocks)):
                    blocks[i] = blocks[i].to(device)

                h_item_batches.append(model.get_repr(blocks))
            h_item = torch.cat(h_item_batches, 0)

        print(h_item.shape)

            # print(evaluation.evaluate_nn(dataset, h_item, args.k, args.batch_size))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random-walk-length', type=int, default=2)
    parser.add_argument('--random-walk-restart-prob', type=float, default=0.5)
    parser.add_argument('--num-random-walks', type=int, default=10)
    parser.add_argument('--num-neighbors', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--hidden-dims', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')  # can also be "cuda:0"
    parser.add_argument('--num-epochs', type=int, default=1)
    parser.add_argument('--batches-per-epoch', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--run-name', type=str, default="training_GCN" + datetime.datetime.now().strftime("%d-%m %H_%M"))
    args = parser.parse_args()

    main(args)
