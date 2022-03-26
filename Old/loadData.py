import torch


#Imports
import pandas as pd

#Drawing data from the user table into a df
u_cols =  ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('../ml-100k/u.user', sep='|', names=u_cols, encoding='latin-1')

#Ratings file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('../ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')

#Items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
items = pd.read_csv('../ml-100k/u.item', sep='|', names=i_cols,
                    encoding='latin-1')


#Get both train and test data files for ratings(test:train = 1:9)
r_cols = ['user_id','movie_id','rating','unix_timestamp']
ratings_train = pd.read_csv('../ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('../ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')
print(ratings_test.columns)
"""
- Load User data + Items + ratings
- Create a graph using this data
- Train a small GCN on this data
- Also Train a CNN which tries to gen embeddings
- Compare the CNN and GCN's performance
"""