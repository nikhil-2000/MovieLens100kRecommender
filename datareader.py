import pandas as pd
from tqdm import tqdm
import numpy as np

base_path = "D:\My Docs/University\Year 4\Individual Project\MovieLens100kRecommender\ml-100k"
user_path = base_path + "/u.user"
item_path = base_path + "/u.item"

class Datareader:

    def __init__(self, path, size=0, training_frac = 1):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user_df = pd.read_csv(user_path, sep='|', names=u_cols, encoding='latin-1')
        self.user_df.set_index("user_id", inplace=True)

        i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date', 'IMDb_URL', 'Unknown', 'Action',
                  'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.items_df = pd.read_csv(item_path, sep='|', names=i_cols,
                                    encoding='latin-1')
        self.items_df.set_index("movie_id", inplace=True)

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_df = pd.read_csv(base_path + "/" + path, sep='\t', names=r_cols, encoding='latin-1')
        self.ratings_df = self.ratings_df.sample(frac = 1)


        if size > 0:
            self.ratings_df = self.ratings_df.head(size)

        self.movie_ids = self.ratings_df.movie_id.unique()
        self.user_ids = self.ratings_df.user_id.unique()

        if size > 0:
            self.items_df = self.items_df[self.items_df.index.isin(self.movie_ids)]
            self.user_df = self.user_df[self.user_df.index.isin(self.user_ids)]

        self.movie_ids = pd.Series(self.movie_ids)
        self.user_ids = pd.Series(self.user_ids)

        train_len = int(training_frac * len(self.items_df))
        test_len = len(self.items_df) - train_len

        self.train = self.ratings_df.head(train_len)
        self.test = self.ratings_df.tail(test_len)

