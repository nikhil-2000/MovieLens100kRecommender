import pandas as pd
from tqdm import tqdm
import numpy as np
from helper_funcs import normalise, categories
from sklearn.model_selection import train_test_split


base_path = "D:\My Docs/University\Year 4\Individual Project\MovieLens100kRecommender\ml-100k"
user_path = base_path + "/u.user"
item_path = base_path + "/u.item"

class Datareader:

    def __init__(self, path, size=0, training_frac = 1, val_frac = 0):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user_df = pd.read_csv(user_path, sep='|', names=u_cols, encoding='latin-1')
        self.user_df.set_index("user_id", inplace=True)

        i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date', 'IMDb_URL'] + categories
        self.items_df = pd.read_csv(item_path, sep='|', names=i_cols,
                                    encoding='latin-1')
        self.items_df.set_index("movie_id", inplace=True)

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_df = pd.read_csv(base_path + "/" + path, sep='\t', names=r_cols, encoding='latin-1')
        self.ratings_df = self.ratings_df.sample(frac = 1)
        self.ratings_df.rating = normalise(self.ratings_df.rating)


        if size > 0:
            self.ratings_df = self.ratings_df.head(size)

        self.movie_ids = self.ratings_df.movie_id.unique()
        self.user_ids = self.ratings_df.user_id.unique()

        if size > 0:
            self.items_df = self.items_df[self.items_df.index.isin(self.movie_ids)]
            self.user_df = self.user_df[self.user_df.index.isin(self.user_ids)]

        self.movie_ids = pd.Series(self.movie_ids)
        self.user_ids = pd.Series(self.user_ids)


        if training_frac < 1:
            self.train, self.test = train_test_split(self.ratings_df, test_size=1 - training_frac)
        else:
            self.train = self.ratings_df

        if val_frac > 0:
            self.train, self.validation = train_test_split(self.train, test_size=val_frac)

