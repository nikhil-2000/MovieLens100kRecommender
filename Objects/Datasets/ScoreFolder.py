import torch

from Objects.Datasets.Data_reader import Dataset
import pandas as pd


class ScoreFolder:
    def __init__(self, filename, include_categories = True, include_metrics = True, size = 0):
        self.dataset = Dataset(filename, size=size)
        self.ratings_df = self.dataset.ratings_df
        self.user_df = self.dataset.user_df

        if include_metrics:
            self.dataset.add_metrics()


        self.movies_df = self.dataset.items_df
        self.movie_ids = self.dataset.movie_ids

        self.include_categories = include_categories
        self.include_metrics = include_metrics

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        # Extracts Categories + Metrics
        movie_id = self.movie_ids.iloc[index]
        s = self.movies_df.loc[movie_id]
        data = self.extract_values(s)

        categories = s.to_list()[4:-6]


        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return s["movie_title"], torch.Tensor(data), categories, movie_id

    def extract_values(self, s: pd.Series):

        # s = s.to_dict()
        # Extracts Categories + Metrics
        data = s.to_list()[4:]

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])
        return torch.Tensor(data)
