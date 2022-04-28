# Aim to generate good embeddings for
#Triplet loss, a triplet = (film, another film that the user watched, a film that the user hasn't watched)
from datetime import datetime
from itertools import product

import torch.nn as nn
import torch
from prettytable import PrettyTable
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from pytorch_metric_learning import miners, losses, distances

from Datasets.Training import TrainDataset
from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork
from Models.GCN.run_eval import test_model, visualise
from datareader import Datareader

torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

# np.random.seed(T_G_SEED)
# torch.manual_seed(T_G_SEED)
# random.seed(T_G_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def output_device():
    if device.type == "cuda":
        print('Using GPU device: ' + torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print('Using CPU device.')


def learn(argv):
    # <batch size> <num epochs> <margin> <learning_rate> <output_name> <input file>
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


    filename = argv[5]


    print('Triplet embeddings training session. Inputs: ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)
    #
    # print("Validation will happen ? ", doValidation)

    datareader = Datareader(filename, size=00000, training_frac=1, val_frac=0.2)

    train_ds = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    val_ds = TrainDataset(datareader.validation, datareader.user_df, datareader.items_df)
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=0)

    # Allow all parameters to be fit
    model = EmbeddingNetwork()

    # model = torch.jit.script(model).to(device) # send model to GPU
    isParallel = torch.cuda.is_available()
    if isParallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=lr)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    easy = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distances.CosineSimilarity())
    semi_hard = miners.TripletMarginMiner(margin=margin, type_of_triplets="semi-hard", distance=distances.CosineSimilarity())
    hard = miners.TripletMarginMiner(margin=margin, type_of_triplets="hard", distance=distances.CosineSimilarity())
    all_miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="all", distance=distances.CosineSimilarity())
    loss_func = losses.TripletMarginLoss(margin=margin)
    # let invalid epochs pass through without training
    if numepochs < 1:
        numepochs = 0
        loss = 0

    run_name = outpath
    train_writer = SummaryWriter(log_dir="runs/" + run_name + "_train")
    val_writer = SummaryWriter(log_dir="runs/" + run_name + "_val")

    train_steps = 0
    val_steps = 0
    steps = 0
    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances
        percent = epoch / numepochs
        # if percent < 0.10:
        #     miner = easy
        # elif 0.10 <= percent <= 0.30:
        #     miner = semi_hard
        # else:
        #     miner = hard
        miner = all_miner

        for phase in ["train", "validation"]:
            epoch_losses = []

            if phase == "train":
                model.train()
                loader = train_loader
                writer = train_writer
                val_steps = steps
                steps = train_steps
            else:
                model.eval()
                loader = val_loader
                writer = val_writer
                train_steps = steps
                steps = val_steps


            for step, (features, labels) in enumerate(
                    tqdm(loader, leave=True, position=0)):


                if phase == "train": optimizer.zero_grad()
                features = features.to(device)  # send tensor to GPU

                embeddings = model(features)
                # Clears space on GPU I think
                del features
                pairs = miner(embeddings, labels)
                # Triplet Loss !!! + Backprop
                loss = loss_func(embeddings, labels, pairs)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                if phase == "train":
                    steps += batch
                else:
                    steps += batch * 4

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




if __name__ == '__main__':

    #<batch size> <num epochs> <margin> <learning_rate> <output_name> <input file>
    train_table = PrettyTable()
    train_table.field_names = ["Params", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    validation_table = PrettyTable()
    validation_table.field_names = ["Params", "Mean Reciprocal Rank", "Average Precision", "Recall By User"]

    batches = [2048]#[64, 128, 256]
    epochs = [10]#[30, 40]
    margins = [0.05, 0.1,0.5,1]
    lrs = [0.001] #[0.01, 0.1, 0.5, 1]
    input = ["ua.base"]

    params = [batches, epochs, margins, lrs, input]

    for ps in product(*params):
        b, e, m, l, i = ps
        s = [str(x) for x in ps]
        model_name =  "_".join(s) + "_" + datetime.now().strftime("%b%d_%H-%M-%S")
        model_path = "WeightFiles/" + model_name
        learn([b, e , m ,l , model_path, i])

        model_file = model_name + "pth"
        train_row, val_row = test_model(model_path+".pth")

        train_table.add_row([model_file] + [str(r) for r in train_row])
        validation_table.add_row([model_file] + [str(r) for r in val_row])
        print(train_table)
        print()
        print(validation_table)
        print()
        with open("results.txt", "w") as f:
            f.write(str(train_table) + "\n" + str(validation_table))


        visualise(model_path+".pth", model_name+"_Embedding")


