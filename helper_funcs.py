import numpy as np
from tqdm import tqdm
vector_features = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
                   'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
                   'War', 'Western', "avg_rating", "views", "male_views", "female_views", "occupation", "avg_age"]

categories = ['Unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary',
              'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
              'War', 'Western']

def add_metrics(ratings, users, items):
    # A sample can have avg rating, watch count, male viewers, female viewers, most common job, average age
    avg_ratings = []
    watch_count = []
    male_count = []
    female_count = []
    most_common_jobs = []
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

        mode_job = 1  # movie_users['occupation'].value_counts().argmax()
        most_common_jobs.append(mode_job)

    items["avg_rating"] = avg_ratings
    items["views"] = watch_count
    items["male_views"] = male_count
    items["female_views"] = female_count
    items["occupation"] = most_common_jobs
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


