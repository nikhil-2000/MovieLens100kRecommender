import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Datasets.GraphDataset_Old import GCNDataset
from Models.GCN.GCNModel import GCNModel
from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork


class CalcEmbeddings:

    def __init__(self, dataloader, model_file, graphDataset : GCNDataset):
        self.model_file = model_file
        self.dataloader = dataloader
        self.set_model()
        self.dataset = graphDataset


    def set_model(self):
        model = GCNModel()
        checkpoint = torch.load(self.model_file)
        model.load_state_dict(checkpoint['model_state_dict'])

        self.model = model

    def get_embeddings(self):
        embeddings = []
        embeddings_dict = {}
        metadata = []
        print("\nGenerating Embeddings")
        # for vector, info in tqdm(self.dataloader):
        for mini_batch in tqdm(self.dataloader):

            input_nodes, output_nodes, blocks = mini_batch
            batch_inputs = blocks[0].srcdata['features']
            out = self.model(blocks, batch_inputs)["movie"]
            out = out.detach().numpy()
            embeddings.extend(out)

            graph_ids = blocks[-1].dstdata['_ID']['movie'].tolist()
            movie_ids = [self.dataset.graph_id_to_movie[_id] for _id in graph_ids]
            metadata.extend(movie_ids)
            keys = movie_ids
            for i,k in enumerate(keys):
                k = int(k)
                embeddings_dict[k] = out[i]


        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
