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
        self.ratings_df = self.ratings_df.sample(frac = 1, random_state=1)
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
            self.train, self.test = train_test_split(self.ratings_df, test_size=1 - training_frac, random_state=1)
        else:
            self.train = self.ratings_df

        if val_frac > 0:
            self.train, self.validation = train_test_split(self.train, test_size=val_frac, random_state=1)



def name_to_id(names, item_df):

    return [item_df.index[item_df.movie_title == name][0].item() for name in names]

if __name__ == '__main__':
    d = Datareader("u.data",0,training_frac=1, val_frac=0.25)

    df = d.ratings_df.groupby("movie_id").size()
    gcn_names = ['Contact (1997)', 'Air Force One (1997)', 'English Patient, The (1996)', 'Cop Land (1997)', 'Scream (1996)', 'Titanic (1997)', 'Liar Liar (1997)', 'Saint, The (1997)', 'Evita (1996)']
    gcn_ids = name_to_id(gcn_names, d.items_df)
    gcn_counts = df.loc[gcn_ids].tolist()

    nn_ids = name_to_id(['Mad City (1997)', 'Jean de Florette (1986)', '8 1/2 (1963)', "Ulee's Gold (1997)", 'Manon of the Spring (Manon des sources) (1986)', 'In the Name of the Father (1993)', 'Casino (1995)', 'Quiz Show (1994)', 'Field of Dreams (1989)'], d.items_df)
    nn_counts = df.loc[nn_ids].tolist()

    pes_ids = name_to_id(['Postino, Il (1994)', 'Good Will Hunting (1997)', 'Butch Cassidy and the Sundance Kid (1969)', 'Titanic (1997)', 'Toy Story (1995)', 'Heathers (1989)', 'Seven (Se7en) (1995)', 'Big Night (1996)', 'Rainmaker, The (1997)'],d.items_df)
    pes_counts = df.loc[pes_ids].tolist()


    import numpy as np

    counts = [gcn_counts, nn_counts, pes_counts]
    for c in counts:
        print(c)
        print(np.mean(c))