from __future__ import absolute_import

import os
import random
import sys

from tqdm import tqdm

project_path = os.path.abspath("../..")
sys.path.insert(0, project_path)

import numpy as np
from PIL import Image
from torchvision.datasets import ImageFolder
import torch
import pandas as pd
from typing import Any, Callable, Optional, Tuple


##Generating Triples method from this article https://towardsdatascience.com/image-similarity-using-triplet-loss-3744c0f67973
# A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MoviesDataset:

    def __init__(self,filename, include_categories = True, include_metrics = True, size = 0 ):

        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user_df = pd.read_csv('../ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        self.user_df.set_index("user_id", inplace=True)

        i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.movies_df = pd.read_csv('../ml-100k/u.item', sep='|', names=i_cols,
                                    encoding='latin-1')
        self.movies_df.set_index("movie_id", inplace=True)

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_df = pd.read_csv(filename, sep='\t', names=r_cols, encoding='latin-1')


        if size > 0:
            self.ratings_df = self.ratings_df.head(size)

        self.movie_ids = self.ratings_df.movie_id.unique()
        self.user_ids = self.ratings_df.user_id.unique()

        if size > 0:
            self.movies_df = self.movies_df[self.movies_df.index.isin(self.movie_ids)]
            self.user_df = self.user_df[self.user_df.index.isin(self.user_ids)]


        if include_metrics:
            self.add_metrics()

        cols_to_norm = self.movies_df.columns[5::]
        # self.movies_df[cols_to_norm] = self.movies_df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        self.include_metrics = include_metrics
        self.include_categories = include_categories

    def __len__(self):
        return len(self.movie_ids)


    def __getitem__(self, index):
        current_movie = self.items.loc[index]
        movie_id = current_movie.loc['movie_id']
        users_who_watched_movie = self.ratings[self.ratings.movie_id == movie_id]

        random_user = users_who_watched_movie.sample()
        positive, negative = self.get_pair(random_user, current_movie)
        # positive = self.items.loc[self.items["movie_id"] == positive_id]
        # negative = self.items.loc[self.items["movie_id"] == negative_id]

        samples = [current_movie,positive.squeeze(), negative.squeeze()]
        return [self.extract_values(s) for s in samples]

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

    def get_pair(self,user, anchor):
        user_id = user.get("user_id").item()
        user_ratings = self.ratings.loc[self.ratings["user_id"] == user_id]

        good_ratings, bad_ratings = user_ratings.loc[user_ratings["rating"] > 2.5], user_ratings.loc[user_ratings["rating"] < 2.5]


        good_filter = self.items['movie_id'].isin(good_ratings['movie_id'])
        good_movies = self.items[good_filter]
        good_movie_vectors = [(i,self.extract_values(row)) for i,row in good_movies.iterrows() if row["movie_id"] != anchor["movie_id"]]

        bad_filter = self.items['movie_id'].isin(bad_ratings['movie_id'])
        bad_movies = self.items[bad_filter]
        bad_movie_vectors = [(i,self.extract_values(row)) for i, row in bad_movies.iterrows() if
                              row["movie_id"] != anchor["movie_id"]]

        anchor_vec = self.extract_values(anchor)

        if len(good_movie_vectors) == 0:
            positive = self.items.sample()
        else:
            diffs = [(i, abs(v - anchor_vec)) for i, v in good_movie_vectors]
            distances = [(i, torch.norm(v)) for i, v in diffs]
            positive_idx, _ = min(distances, key=lambda t: t[1])
            positive = self.items.iloc[positive_idx]


        if len(bad_movie_vectors) == 0:
            negative = self.items.sample()
        else:
            diffs = [(i,abs(v - anchor_vec)) for i,v in bad_movie_vectors]
            distances = [(i,torch.norm(v)) for i,v in diffs]
            negative_idx,_ = min(distances, key = lambda t: t[1])
            negative = self.items.iloc[negative_idx]


        return positive,negative

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






class ScoreFolder(MoviesDataset):
    def __init__(self, filename, include_categories = True, include_metrics = True, size = 0):
        super(ScoreFolder, self).__init__(filename, include_categories, include_metrics, size = size)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        # Extracts Categories + Metrics
        movie_id = self.movie_ids[index]
        s = self.movies_df.loc[movie_id]
        data = super().extract_values(s)

        categories = s.to_list()[5:-5]


        # return torch.Tensor([s["avg_rating"],s["views"],s["male_views"],s["female_views"],s["avg_age"]])

        return s["movie_title"], torch.Tensor(data), categories, movie_id


def show_example_triplet(triple):
    anchor_im, positive_im, negative_im = triple

    dst = Image.new('RGB', (anchor_im.width + positive_im.width + negative_im.width, anchor_im.height))
    dst.paste(anchor_im, (0, 0))
    dst.paste(positive_im, (anchor_im.width, 0))
    dst.paste(negative_im, (anchor_im.width + positive_im.width, 0))
    dst.show()


def getListImages(images):
    h, w = images[0].height, images[0].width
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0
    for i in images:
        dst.paste(i, (x, y))
        x += w

    return dst


def showImages(images, size=(100, 100)):
    h, w = size
    dst = Image.new('RGB', (len(images) * w, h))
    x = 0
    y = 0

    for i in images:
        i = i.resize(size)
        dst.paste(i, (x, y))
        x += w

    dst.show()


def tensor_to_image(tensor):
    tensor = tensor * 255
    c, w, h = tensor.shape
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor, 'RGB')


if __name__ == '__main__':

    # dataset = old.ClothesFolder(images_path, transform = transform)
    dataset_net = MoviesDataset()
    print(dataset_net[10])
    # print(test)
