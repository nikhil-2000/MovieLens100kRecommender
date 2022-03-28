import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Objects.Models.NeuralNetworkModel import EmbeddingNetwork
from Objects.Datasets.ScoreFolder import ScoreFolder


class CalcEmbeddings:

    def __init__(self, data_file, model_file, size = 0):
        self.model_file = model_file
        self.set_model()
        self.dataset = ScoreFolder(data_file, size = size)
        self.dataloader = DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=0)


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
        for title, vector, categories, movie_id in tqdm(self.dataloader):
            out = self.model(vector)
            out = out.detach().numpy()
            embeddings.append(out)

            # title = title.detach()
            # skill_id = int(skill_id.detach().numpy())
            # skill_name = skill_name[0]
            movie_id = movie_id.detach().numpy()[0]
            metadata.append(movie_id)
            embeddings_dict[movie_id] = out


        embeddings = np.array(embeddings).squeeze()

        return embeddings, metadata, embeddings_dict
