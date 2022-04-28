import random

import torch

from Datasets.DatasetBase import DatasetBase
from datareader import Datareader
from helper_funcs import categories,vector_features,add_metrics


class TrainDataset(DatasetBase):
    def __init__(self, interactions, users, items):
        # self.dataset = Dataset(filename, size=size)
        super(TrainDataset, self).__init__(interactions, users, items)


    def __getitem__(self, index: int):
        row = self.interaction_df.iloc[index]
        # Extracts Categories + Metrics
        movie_id = row.movie_id
        user_id = row.user_id
        user_rating = self.users[self.id_to_idx[user_id]].avg_rating
        s = self.item_df.loc[movie_id]
        movie_data = s[vector_features].to_list()
        user_data = [user_rating, row.rating]

        data = user_data + movie_data

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return torch.Tensor(data), user_id

    def get_movie_data(self, movie_id):
        # Extracts Categories + Metrics
        s = self.item_df.loc[movie_id]
        movie_data = s[vector_features].to_list()
        vector = torch.Tensor(movie_data)
        lbl = (vector == 1).nonzero()[0]


        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return vector, lbl


    def reduce_users_films(self):
        u_ids = self.interaction_df.user_id.unique()
        i_ids = self.interaction_df.movie_id.unique()
        users = self.user_df.loc[self.user_df.index.isin(u_ids)]
        items = self.item_df.loc[self.item_df.index.isin(i_ids)]

        self.user_df = users
        self.item_df = items
        self.item_ids = self.item_df.index.to_series()


if __name__ == '__main__':
    datareader = Datareader("ua.base")
    mt = TrainDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)
    print(mt[0])
