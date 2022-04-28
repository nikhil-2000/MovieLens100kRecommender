import argparse
import datetime
from itertools import product

import dgl
import numpy as np
import torch
from dgl.dataloading import NeighborSampler
from prettytable import PrettyTable
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm

from Datasets.GraphDataset_Old import GCNDataset
from Datasets.Training import TrainDataset
from Models.GCN.GCNModel import GCNModel
from datareader import Datareader
from helper_funcs import categories
from pytorch_metric_learning import losses, miners, distances
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def learn(argv, datareader):
    usagemessage = "Should be 'python train_NN.py <batch size> <num epochs> <margin> <output_name>'"
    if len(argv) < 4:
        print(usagemessage)
        return

    batch = int(argv[0])
    assert batch > 0, "Batch size should be more than 0\n" + usagemessage

    numepochs = int(argv[1])
    assert numepochs > 0, "Need more than " + str(numepochs) + " epochs\n" + usagemessage

    margin = float(argv[2])
    assert 0 < margin, "Pick a margin greater than 0\n" + usagemessage

    lr = float(argv[3])

    outpath = argv[4]


    # load and preprocess dataset
    trainDataset = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
    valDataset = TrainDataset(datareader.validation, datareader.user_df, datareader.items_df)
    trainGraphDataset = GCNDataset(trainDataset)
    valGraphDataset = GCNDataset(valDataset)


    train_nids = {"movie": trainGraphDataset.graph_movie_ids}
    sampler = NeighborSampler([20,20])  # create a sampler
    train_loader = dgl.dataloading.DataLoader(
        trainGraphDataset.graph,
        train_nids,
        sampler,
        batch_size=batch_size,  # batch_size decides how many IDs are passed to sampler at once
        # other arguments
    )

    val_nids = {"movie": valGraphDataset.graph_movie_ids}
    validation_loader = dgl.dataloading.DataLoader(
        valGraphDataset.graph,
        val_nids,
        sampler,
        batch_size=batch_size,  # batch_size decides how many IDs are passed to sampler at once
        # other arguments
    )

    model = GCNModel()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    isParallel = False

    all_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distances.CosineSimilarity())
    loss_func = losses.TripletMarginLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    run_name = outpath
    train_writer = SummaryWriter(log_dir="runs/" + run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

    train_steps = 0
    val_steps = 0
    steps = 0

    for epoch in tqdm(range(20), desc="Epochs"):

        miner = all_miner

        for phase in ["train", "val"]:
            epoch_losses = []

            if phase == "train":
                model.train()
                loader = train_loader
                writer = train_writer
                val_steps = steps
                steps = train_steps
            else:
                model.eval()
                loader = validation_loader
                writer = val_writer
                train_steps = steps
                steps = val_steps

            for step, mini_batch in enumerate(
                    tqdm(loader, leave=True, position=0)):

                if phase == "train": optimizer.zero_grad()

                input_nodes, output_nodes, blocks = mini_batch
                batch_inputs = blocks[0].srcdata['features']
                labels = blocks[-1].dstdata["label"]["movie"].squeeze()
                embeddings = model(blocks, batch_inputs)["movie"]

                if phase == "train":
                    steps += step
                else:
                    steps += step * (1/0.3)

                pairs = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, pairs)

                writer.add_scalar("triplet_loss/", loss, steps)
                writer.add_scalar("pairs/", len(pairs[0]), steps)
                epoch_losses.append(loss.cpu().detach().numpy())

            # batch_norm = torch.linalg.norm(anchor_out, ord = 1, dim= 1)
            # embedding_norm = torch.mean(batch_norm)
            # train_writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

            writer.add_scalar("Epoch_triplet_loss/", np.mean(epoch_losses), epoch + 1)

        weights = model.module.state_dict() if isParallel else model.state_dict()
        torch.save({
            'epoch': epoch,
            'model_state_dict': weights,
            'optimzier_state_dict': optimizer.state_dict(),
            'loss': loss
        }, outpath + '.pth')



def evaluate(args, model, dataset):
    pass



if __name__ == '__main__':

    datareader = Datareader("ua.base", training_frac=1, val_frac=0.3)
    ps = [64,20,0.2,0.001]
    batch_size, epochs, margin, learn_rate = ps
    s = [str(x) for x in ps]
    model_name = "_".join(s) + "_" + datetime.datetime.now().strftime("%b%d_%H-%M-%S")
    model_path = "WeightFiles/" + model_name
    learn([batch_size, epochs, margin, learn_rate, model_path], datareader)

    # model_file = model_name + "pth"
    # train_row, val_row = test_model(model_path + ".pth")
    #
    # train_table.add_row([model_file] + [str(r) for r in train_row])
    # validation_table.add_row([model_file] + [str(r) for r in val_row])
    # print(train_table)
    # print()
    # print(validation_table)
    # print()
    # with open("results.txt", "w") as f:
    #     f.write(str(train_table) + "\n" + str(validation_table))

