import os
import random

from prettytable import PrettyTable
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from Datasets.Testing import TestDataset
from Models.MetricBase import MetricBase
from Models.NeuralNetwork.compute_embeddings import CalcEmbeddings
from datareader import Datareader
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

import random

from helper_funcs import headers


class NormalNNMetrics(MetricBase):

    def __init__(self, dataloader, model_file, dataset, train_embeddings_metadata = None):
        self.embedder = CalcEmbeddings(dataloader, model_file)
        self.dataset = dataset
        self.generated_embeddings = False

        if train_embeddings_metadata == None:
            self.embeddings, self.metadata, self.e_dict = self.embedder.get_embeddings()
            self.train_embeddings = self.embeddings
            self.metadata = pd.DataFrame(self.metadata, columns=headers)
            self.generated_embeddings = True

        else:
            embs, metadata = train_embeddings_metadata
            self.train_embeddings = embs
            self.metadata = pd.DataFrame(metadata, columns=headers)
        
        super(NormalNNMetrics, self).__init__()



    def top_n_items(self, anchor, search_size):
        # df_metadata = pd.DataFrame(metadata, columns=["problem_id", "skill_id", "skill_name"])
        key = (anchor.movie_id.item(), anchor.user_id.item())
        if not self.generated_embeddings:
            vector = self.dataset.get_by_ids(key[0],key[1])
            anchor_embedding = self.embedder.model(vector).detach().numpy()
        else:
            anchor_embedding = self.e_dict[key]

        dists = cosine_dists(anchor_embedding, self.train_embeddings) #np.linalg.norm(self.embeddings - anchor_embedding, axis=1)
        sorted_indexes = np.argsort(dists)
        # best_indexes = sorted_indexes[1:search_size + 1]
        sorted_ids = self.metadata.iloc[sorted_indexes].movie_id.drop_duplicates().tolist()

        if len(sorted_ids) >= search_size:
            return sorted_ids[:search_size]
        else:
            return sorted_ids + [0] * (search_size - len(sorted_ids))



    def rank_questions(self, ids, anchor):
        anchor_id = anchor.movie_id.item()
        anchor = self.e_dict[anchor_id]

        embeddings_to_rank = [self.e_dict[k] for k in ids]
        embeddings_to_rank = np.array(embeddings_to_rank).squeeze()

        dists = np.linalg.norm(embeddings_to_rank - anchor, axis=1)
        sorted_indexes = np.argsort(dists)

        return np.array(ids)[sorted_indexes].tolist()

    def average_distances(self, anchor_id, positive_ids, negative_ids):

        anchor = self.e_dict[anchor_id]

        positives = [self.e_dict[k] for k in positive_ids]
        positives = np.array(positives).squeeze()


        negatives = [self.e_dict[k] for k in negative_ids]
        negatives = np.array(negatives).squeeze()

        pos_dists = np.linalg.norm(positives - anchor, axis=1)
        neg_dists = np.linalg.norm(negatives - anchor, axis=1)


        return np.mean(pos_dists), np.mean(neg_dists)


def cosine_dists(u, vs):
    u_dot_v = np.sum(u * vs, axis=1)

    # find the norm of u and each row of v
    mod_u = np.sqrt(np.sum(u * u))
    mod_v = np.sqrt(np.sum(vs * vs, axis=1))

    # just apply the definition
    final = 1 - u_dot_v / (mod_u * mod_v)
    return final
