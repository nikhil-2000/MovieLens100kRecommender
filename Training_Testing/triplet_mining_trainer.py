# Aim to generate good embeddings for
#Triplet loss, a triplet = (film, another film that the user watched, a film that the user hasn't watched)
import datetime

import torch.nn as nn
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Objects.Datasets.MinerTrainer import TrainDataset
from Objects.Models.NeuralNetworkModel import EmbeddingNetwork
import numpy as np
import sys
from pytorch_metric_learning import miners, losses



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
    # <batch size> <num epochs> <margin> <output_name>
    argv = argv[1:]
    usagemessage = "Should be 'python train_NN.py <batch size> <num epochs> <margin> <output_name>'"
    if len(argv) < 4:
        print(usagemessage)
        return


    batch = int(argv[0])
    assert batch > 0, "Batch size should be more than 0\n" + usagemessage

    numepochs = int(argv[1])
    assert numepochs > 0, "Need more than " + str(numepochs) + " epochs\n" + usagemessage

    outpath = argv[3] + "_" + datetime.datetime.now().strftime("%b%d_%H-%M-%S")

    margin = float(argv[2])
    assert 0 < margin, "Pick a margin greater than 0\n" + usagemessage

    filename = argv[4]

    phases = ["train"]

    print('Triplet embeddings training session. Inputs: ' + str(
        batch) + ', ' + str(numepochs) + ', ' + str(margin) + ', ' + outpath)
    #
    # print("Validation will happen ? ", doValidation)


    train_ds = TrainDataset(filename)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)

    datasets = {"train": train_ds}
    data_loaders = {"train": train_loader}

    # Allow all parameters to be fit
    model = EmbeddingNetwork(in_channels=25, out_channels=300)

    # model = torch.jit.script(model).to(device) # send model to GPU
    isParallel = torch.cuda.is_available()
    if isParallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model = model.to(device)  # send model to GPU

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # criterion = torch.jit.script(TripletLoss(margin=10.0))
    miner = miners.MultiSimilarityMiner(epsilon=margin)
    loss_func = losses.TripletMarginLoss(margin=margin)
    # let invalid epochs pass through without training
    if numepochs < 1:
        numepochs = 0
        loss = 0

    run_name = datetime.datetime.now().strftime("%b%d_%H-%M-%S") + "_Epochs" + str(numepochs) + "_Datasize" + str(
        len(train_ds))
    writer = SummaryWriter(log_dir="runs/" + run_name)

    steps = {"train": 0, "validation": 0}
    for epoch in tqdm(range(numepochs), desc="Epochs"):
        # Split data into "Batches" and calc distances

        for phase in phases:


            dataset, data_loader = datasets[phase], data_loaders[phase]

            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            epoch_losses = []
            for step, (features, labels) in enumerate(
                    tqdm(data_loader, desc=phase, leave=True, position=0)):
                optimizer.zero_grad()
                features = features.to(device)  # send tensor to GPU

                embeddings = model(features)
                # Clears space on GPU I think
                del features

                # Triplet Loss !!! + Backprop
                hard_pairs = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_pairs)
                loss.backward()
                optimizer.step()

                # if phase == 'train':
                #     loss.backward()
                #     optimizer.step()

                epoch_losses.append(loss.cpu().detach().numpy())

                # batch_norm = torch.linalg.norm(anchor_out, ord = 1, dim= 1)
                # embedding_norm = torch.mean(batch_norm)
                # writer.add_scalar("Loss/embedding_norm", embedding_norm, s)

                writer.add_scalar("triplet_loss/" + phase, loss, steps[phase])

                steps[phase] += batch

            writer.add_scalar("Epoch_triplet_loss/" + phase, np.mean(epoch_losses), epoch + 1)

            print("Epoch: {}/{} - Loss: {:.4f}".format(epoch + 1, numepochs, np.mean(epoch_losses)))

            # Saves model so that distances can be updated using new model

            if phase == "train":
                weights = model.module.state_dict() if isParallel else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': weights,
                    'optimzier_state_dict': optimizer.state_dict(),
                    'loss': loss
                }, outpath + '.pth')

                dataset.modelfile = outpath + '.pth'

    epoch += 1


if __name__ == '__main__':

    learn(sys.argv)