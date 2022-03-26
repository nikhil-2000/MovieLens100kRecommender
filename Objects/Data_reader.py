import pandas as pd
from tqdm import tqdm
import numpy as np

class Dataset:

    def __init__(self, path, size=0):
        u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
        self.user_df = pd.read_csv('../ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')
        self.user_df.set_index("user_id", inplace=True)

        i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date', 'IMDb_URL', 'unknown', 'Action',
                  'Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.items_df = pd.read_csv('../ml-100k/u.item', sep='|', names=i_cols,
                                    encoding='latin-1')
        self.items_df.set_index("movie_id", inplace=True)

        r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
        self.ratings_df = pd.read_csv(path, sep='\t', names=r_cols, encoding='latin-1')



        if size > 0:
            self.ratings_df = self.ratings_df.head(size)

        self.movie_ids = self.ratings_df.movie_id.unique()
        self.user_ids = self.ratings_df.user_id.unique()

        if size > 0:
            self.items_df = self.items_df[self.items_df.index.isin(self.movie_ids)]
            self.user_df = self.user_df[self.user_df.index.isin(self.user_ids)]

        self.movie_ids = pd.Series(self.movie_ids)
        self.user_ids = pd.Series(self.user_ids)


    def __len__(self):
        return len(self.movie_ids)


    def add_metrics(self):
        # A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
        avg_ratings = []
        watch_count = []
        male_count = []
        female_count = []
        most_common_jobs = []
        avg_ages = []
        print("Adding Metrics")
        for i, row in tqdm(self.items_df.iterrows(), total=len(self.items_df)):
            movie_id = i
            movie_ratings = self.ratings_df.loc[self.ratings_df.movie_id == movie_id]
            avg_rating = movie_ratings["rating"].mean()
            avg_ratings.append(avg_rating)
            watch_count.append(len(movie_ratings))
    
            user_filter = self.user_df.index.isin(movie_ratings['user_id'])
            movie_users = self.user_df[user_filter]
    
            m_f_count = movie_users['sex'].value_counts()
            male_count.append(m_f_count.get("M", 0))
            female_count.append(m_f_count.get("F", 0))
    
            avg_age = movie_users["age"].mean()
            avg_ages.append(avg_age)
    
            mode_job = 1  # movie_users['occupation'].value_counts().argmax()
            most_common_jobs.append(mode_job)
    
        self.items_df["avg_rating"] = avg_ratings
        self.items_df["views"] = watch_count
        self.items_df["male_views"] = male_count
        self.items_df["female_views"] = female_count
        self.items_df["occupation"] = most_common_jobs
        self.items_df["avg_age"] = avg_ages

        self.items_df.replace({np.nan: 0})