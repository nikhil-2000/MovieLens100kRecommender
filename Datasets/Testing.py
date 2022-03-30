import torch

from helper_funcs import categories, vector_features, add_metrics


class TestDataset:
    def __init__(self, interactions, users, items):
        # self.dataset = Dataset(filename, size=size)
        self.interaction_df = interactions
        self.user_df = users

        self.item_df = add_metrics(interactions, users, items)
        self.item_ids = self.item_df.index.to_series()

    def __len__(self):
        return len(self.item_ids)

    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        movie_id = self.item_ids.iloc[index]
        s = self.item_df.loc[movie_id]
        data = s[vector_features].to_list()

        return torch.Tensor(data), (s["movie_title"],  s[categories].to_list(), movie_id)
