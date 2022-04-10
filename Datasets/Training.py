import random

import torch

from helper_funcs import categories,vector_features,add_metrics


class TrainDataset:
    def __init__(self, interactions, users, items):
        # self.dataset = Dataset(filename, size=size)
        self.interaction_df = interactions
        self.user_df = users

        self.item_df = add_metrics(interactions, users, items)
        self.item_ids = self.item_df.index.to_series()

    def __len__(self):
        return len(self.item_df)

    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        movie_id = self.item_ids.iloc[index]
        s = self.item_df.loc[movie_id]
        data = s[vector_features].to_list()

        potential_cs = s[categories]
        category = self.pick_cat(potential_cs)

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return torch.Tensor(data), category

    def pick_cat(self, cats):
        is_one_idx = [i for i, c in enumerate(cats) if c == 1]
        return random.choice(is_one_idx)

    def reduce_users_films(self):
        u_ids = self.interaction_df.user_id.unique()
        i_ids = self.interaction_df.movie_id.unique()
        users = self.user_df.loc[self.user_df.index.isin(u_ids)]
        items = self.item_df.loc[self.item_df.index.isin(i_ids)]

        self.user_df = users
        self.item_df = items
        self.item_ids = self.item_df.index.to_series()


if __name__ == '__main__':

    mt = TrainDataset("../../../ml-100k/ua.base")
    print(mt[0])
