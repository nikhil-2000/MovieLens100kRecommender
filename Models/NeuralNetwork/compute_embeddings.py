import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork


class CalcEmbeddings:

    def __init__(self, dataloader, model_file):
        self.model_file = model_file
        self.dataloader = dataloader
        self.set_model()


    def set_model(self):
        model = EmbeddingNetwork()
        checkpoint = torch.load(self.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.model = model

    def get_embeddings(self):
        embeddings = []
        embeddings_dict = {}
        metadata = []
        print("\nGenerating Embeddings")
        # for vector, info in tqdm(self.dataloader):
        for vector, rows, keys in self.dataloader:

            rows = torch.stack(rows).detach().T.tolist()
            rows = list(map(tuple, rows))
            out = self.model(vector)
            out = out.detach().numpy()
            embeddings.extend(out)
            metadata.extend(rows)

            keys = torch.stack(keys).detach().T.tolist()
            for i,k in enumerate(keys):
                k = tuple(k)
                embeddings_dict[k] = out[i]


        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
