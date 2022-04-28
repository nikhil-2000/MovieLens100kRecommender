import torch

from Datasets.DatasetBase import DatasetBase
from datareader import Datareader
from helper_funcs import categories, vector_features, add_metrics


class TestDataset(DatasetBase):
    def __init__(self, interactions, users, items):
        # self.dataset = Dataset(filename, size=size)
        super(TestDataset, self).__init__(interactions, users, items)




    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
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
        metadata_row = data + [movie_id, user_id]

        return torch.Tensor(data), metadata_row, (movie_id, user_id)


if __name__ == '__main__':
    datareader = Datareader("ua.base")
    test_data = TestDataset(datareader.ratings_df, datareader.user_df, datareader.items_df)