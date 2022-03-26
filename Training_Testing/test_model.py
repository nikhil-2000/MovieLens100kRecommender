from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from Training_Testing.train_NN import EmbeddingNetwork
import torch
from Old import triplesLoader as tl

model = EmbeddingNetwork()
checkpoint = torch.load("../normal_nn_Mar23_19-08-49.pth")
model.load_state_dict(checkpoint['model_state_dict'])

score_ds = tl.ScoreFolder("../ml-100k/ua.base")
score_loader = DataLoader(score_ds, batch_size=1, shuffle=False, num_workers=1)

results = []
names = []
metadata = []
headers = ['name','unknown', 'Action','Adventure',
                  'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                  'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']


for name,movie,categories,r in score_ds:
    x = model(movie)
    results.append(x)
    names.append(name)
    m = [name] + categories
    metadata.append(tuple(m))

logdir = "runs/embedding_vis"

results = torch.stack(results,dim=1).T
writer = SummaryWriter()
writer.add_embedding(results, metadata=metadata, metadata_header=headers)



print(len(results))