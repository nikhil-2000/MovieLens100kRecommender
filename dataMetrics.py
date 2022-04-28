from datareader import Datareader
import pandas as pd
from helper_funcs import categories

base_path = "D:\My Docs/University\Year 4\Individual Project\MovieLens100kRecommender\ml-100k"
user_path = base_path + "/u.user"
item_path = base_path + "/u.item"


u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
user_df = pd.read_csv(user_path, sep='|', names=u_cols, encoding='latin-1')
# user_df.set_index("user_id", inplace=True)

i_cols = ['movie_id', 'movie_title', 'release_date', 'video_release date', 'IMDb_URL'] + categories
items_df = pd.read_csv(item_path, sep='|', names=i_cols,
                            encoding='latin-1')
# items_df.set_index("movie_id", inplace=True)

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_df = pd.read_csv(base_path + "/" + "u.data", sep='\t', names=r_cols, encoding='latin-1')
ratings_df = ratings_df.sample(frac=1)



user_counts = ratings_df.groupby("user_id").size()
film_counts = ratings_df.groupby("movie_id").size()


def data_analysis(series, name):
    print("-" * 200)
    print()
    print("Ratings Per {} mean: {}".format(name, series.mean()))
    print("Ratings Per {} median: {}".format(name, series.median()))
    # print("Ratings Per {} mode: {}".format(name, seri()))
    print("Ratings Per {} std: {}".format(name, series.std()))
    print("Max # of {} ratings : {}".format(name, series.max()))
    print("Min # of {} ratings : {}".format(name, series.min()))
    print()
    print("-" * 200)


print("Users:", len(user_df))
print("Movies:", len(items_df))
print("Ratings:", len(ratings_df))
data_analysis(user_counts, "user")
data_analysis(film_counts, "movies")
data_analysis(ratings_df.rating, "rating")

print("Movie Cols:", items_df.columns.tolist())
print("User Cols:", user_df.columns.tolist())
print("Rating Cols:", ratings_df.columns.tolist())

user_ex = user_df.head(1).squeeze()
items_ex = items_df.head(1).squeeze()
ratings_ex = ratings_df.head(1).squeeze()

print()