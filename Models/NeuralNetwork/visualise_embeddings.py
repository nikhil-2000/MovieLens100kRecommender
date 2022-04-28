from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter

from Datasets.Testing import TestDataset
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from datareader import Datareader
from helper_funcs import headers
from torch.utils.data import DataLoader


def write_data_to_board(xs, run_name, metadata=[], headers=[]):
    writer = SummaryWriter(log_dir="runs/" + run_name)
    if metadata == [] or headers == "":
        writer.add_embedding(xs)
    else:
        writer.add_embedding(xs, metadata=metadata, metadata_header = headers)
    writer.flush()

def add_embeddings_to_tensorboard(datareader,model_file, file_name):
    train = TestDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
    loader = DataLoader(train, batch_size=64)


    train_embedder = CalcEmbeddings(loader, model_file)
    embeddings, metadata, _ = train_embedder.get_embeddings()

    print("# of Embeddings : ", len(embeddings))
    run_name = datetime.now().strftime("%b%d_%H-%M-%S") + "_" + file_name
    write_data_to_board(embeddings,run_name, metadata=metadata, headers = headers)


if __name__ == '__main__':
    model_file = "WeightFiles/1024_10_0.5_0.001Apr25_12-11-11.pth"
    file = "../../skill_builder_data.csv"
    train_reader = Datareader(file, size=20000, training_frac=0.7, val_frac=0.2)
    add_embeddings_to_tensorboard(train_reader, model_file, "training_data")
