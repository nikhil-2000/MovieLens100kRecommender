import numpy as np
from tqdm import tqdm
vector_features = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                   'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                   'War', 'Western', "avg_rating", "views", "male_views", "female_views", "avg_age"]
# vector_features = ["avg_rating", "views", "male_views", "female_views", "avg_age"]

categories = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']

headers = ["User Average Rating","Rating"] + vector_features + ["movie_id", "user_id"]

def add_metrics(ratings, users, items):
    # A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
    avg_ratings = []
    watch_count = []
    male_count = []
    female_count = []
    avg_ages = []
    print("\nAdding Metrics")
    # for i, row in tqdm(items.iterrows(), total=len(items)):
    for i, row in items.iterrows():
        movie_id = i
        movie_ratings = ratings.loc[ratings.movie_id == movie_id]
        avg_rating = movie_ratings["rating"].mean()
        avg_ratings.append(avg_rating)
        watch_count.append(len(movie_ratings))

        user_filter = users.index.isin(movie_ratings['user_id'])
        movie_users = users[user_filter]

        m_f_count = movie_users['sex'].value_counts()
        male_count.append(m_f_count.get("M", 0))
        female_count.append(m_f_count.get("F", 0))

        avg_age = movie_users["age"].mean()
        avg_ages.append(avg_age)

    items["avg_rating"] = avg_ratings
    items["views"] = watch_count
    items["male_views"] = male_count
    items["female_views"] = female_count
    items["avg_age"] = avg_ages


    items.fillna(0,inplace = True)
    items.replace({np.nan: 0}, inplace = True)
    normalised = normalise(items[vector_features])
    items[vector_features] = normalised[vector_features]
    items.fillna(0, inplace = True)
    items.replace({np.nan: 0}, inplace = True)


    return items

def normalise(df):
    return (df-df.min())/(df.max()-df.min())


def MRR(positive_ids, top_n_rec):

    for i, rec_id in enumerate(top_n_rec):
        if rec_id in positive_ids:
            return 1/(i+1)

    return 0

def AveragePrecision(positive_ids, top_n_rec):
    precision_at_i = []
    positives = 0
    total = 0
    for i, rec_id in enumerate(top_n_rec):
        total += 1
        if rec_id in positive_ids:
            positives += 1

        precision_at_i.append(positives / total)

    return np.mean(precision_at_i)

def Recall(positive_ids, top_n_rec):
    total = len(positive_ids)
    rec = sum([1 for rec_id in top_n_rec if rec_id in positive_ids])
    return rec/total



def convert_distances(distances, search_size):
    x = np.exp(-distances)
    weights = x / x.sum(axis=0)
    x = search_size * weights
    x = np.round(x)
    leftover = search_size - sum(x)
    if leftover > 0:
        i = 0
        while leftover > 0:
            x[i] += 1
            leftover -= 1
            i = (i+1) % search_size
    elif leftover < 0:
        i = len(x) - 1
        while leftover < 0:
            x[i] -= 1
            leftover += 1
            i = (i-1) % search_size


    return x
