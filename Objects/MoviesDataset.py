from __future__ import absolute_import

import os
import random
import sys

from tqdm import tqdm

project_path = os.path.abspath("..")
sys.path.insert(0, project_path)

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import torch
import pandas as pd
from typing import Any, Callable, Optional, Tuple
from Objects.Data_reader import Dataset

class TriplesGenerator:

    def __init__(self,filename, include_categories = True, include_metrics = True, size = 0 ):

        self.dataset = Dataset(filename, size=size)
        self.ratings_df = self.dataset.ratings_df
        self.user_df = self.dataset.user_df
        self.movies_df = self.dataset.items_df


        if include_metrics:
            self.add_metrics()

        self.movies_df.fillna(0)


        cols_to_norm = self.movies_df.columns[5::]
        self.movies_df[cols_to_norm] = self.movies_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        self.include_metrics = include_metrics
        self.include_categories = include_categories

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index):
        movie_id = self.movies_df.index[index]
        current_movie = self.movies_df[movie_id]
        return self.extract_values(s),

        # return current_movie, positive.squeeze(), negative.squeeze()

    def extract_values(self,s : pd.Series):

        # s = s.to_dict()
        #Extracts Categories + Metrics
        if self.include_metrics and self.include_categories:
            data = s.to_list()[5:]
        elif self.include_categories:
        #Extracts Categories
            data = s.to_list()[5:-5]
        else:
        #Extracts Metrics
            data = s.to_list()[-5:]

        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])
        return torch.Tensor(data)

    def add_metrics(self):
        # A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
        avg_ratings = []
        watch_count = []
        male_count = []
        female_count = []
        most_common_jobs = []
        avg_ages = []
        print("Adding Metrics")
        for i, row in tqdm(self.movies_df.iterrows(), total = len(self.movies_df)):
            movie_id = i
            movie_ratings = self.ratings_df.loc[self.ratings_df.movie_id == movie_id]
            avg_rating = movie_ratings["rating"].mean()
            avg_ratings.append(avg_rating)
            watch_count.append(len(movie_ratings))

            user_filter = self.user_df.index.isin(movie_ratings['user_id'])
            movie_users = self.user_df[user_filter]

            m_f_count = movie_users['sex'].value_counts()
            male_count.append(m_f_count.get("M",0))
            female_count.append(m_f_count.get("F",0))

            avg_age = movie_users["age"].mean()
            avg_ages.append(avg_age)

            mode_job = 0 #movie_users['occupation'].value_counts().argmax()
            most_common_jobs.append(mode_job)



        self.movies_df["avg_rating"] = avg_ratings
        self.movies_df["views"] = watch_count
        self.movies_df["male_views"] = male_count
        self.movies_df["female_views"] = female_count
        self.movies_df["occupation"] = most_common_jobs
        self.movies_df["avg_age"] = avg_ages