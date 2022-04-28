import dgl
from dgl import apply_each
from dgl.dataloading import NeighborSampler
from pytorch_metric_learning import miners, losses, distances
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Datasets.Training import TrainDataset
from Models.GCN_tripletLoss.GraphDataset import GCNDataset
from datareader import Datareader
import dgl.nn.pytorch as dglnn
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

file = "ua.test"
datareader = Datareader(file,size = 0)
trainDataset = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
graphDataset = GCNDataset(trainDataset)
train_nids = {"movie":graphDataset.graph_movie_ids}
sampler = NeighborSampler([5,5])  # create a sampler
dataloader = dgl.dataloading.DataLoader(
    graphDataset.graph,
    train_nids,
    sampler,
    batch_size=64,    # batch_size decides how many IDs are passed to sampler at once
    # other arguments
)





model = GCNModel(20,100,20,2,F.relu, 0.5, graphDataset.graph.etypes, max(graphDataset.graph_movie_ids) + 1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


isParallel = False
margin = 0.2
lr = 0.001

run_name = "GCN_test_sampler"
outpath = "GCN_TEST"
all_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distances.CosineSimilarity())
loss_func = losses.TripletMarginLoss(margin=margin)
optimizer = optim.Adam(model.parameters(), lr=lr)

train_writer = SummaryWriter(log_dir="runs/" + run_name + "_train")
val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

train_steps = 0
val_steps = 0
steps = 0

for epoch in tqdm(range(1), desc="Epochs"):
    # Split data into "Batches" and calc distances
    # if percent < 0.10:
    #     miner = easy
    # elif 0.10 <= percent <= 0.30:
    #     miner = semi_hard
    # else:
    #     miner = hard
    miner = all_miner

    for phase in ["train"]:
        epoch_losses = []

        if phase == "train":
            model.train()
            # loader = train_loader
            # writer = train_writer
            # val_steps = steps
            # steps = train_steps
        else:
            model.eval()
            # loader = val_loader
            # writer = val_writer
            # train_steps = steps
            # steps = val_steps

        for step, mini_batch in enumerate(
                tqdm(dataloader, leave=True, position=0)):

            if phase == "train": optimizer.zero_grad()

            input_nodes, output_nodes, blocks = mini_batch
            batch_inputs = blocks[0].srcdata['features']
            labels = blocks[-1].dstdata["label"]["movie"].squeeze()
            embeddings = model(blocks, batch_inputs)["movie"]

            if phase == "train":
                steps += step
            else:
                steps += step * 4

            pairs = miner(embeddings, labels)
            loss = loss_func(embeddings, labels, pairs)

            train_writer.add_scalar("triplet_loss/", loss, steps)
            train_writer.add_scalar("pairs/", len(pairs[0]), steps)
            epoch_losses.append(loss.cpu().detach().numpy())

        # batch_norm = torch.linalg.norm(anchor_out, ord = 1, dim= 1)
        # embedding_norm = torch.mean(batch_norm)
        # train_writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

        train_writer.add_scalar("Epoch_triplet_loss/", np.mean(epoch_losses), epoch + 1)

    weights = model.module.state_dict() if isParallel else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': weights,
        'optimzier_state_dict': optimizer.state_dict(),
        'loss': loss
    }, outpath + '.pth')
