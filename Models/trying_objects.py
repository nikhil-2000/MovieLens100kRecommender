from Datasets.Testing import TestDataset
from Datasets.Training import TrainDataset
from datareader import Datareader
from Models.NeuralNetwork.NeuralNetworkModel import EmbeddingNetwork
from torch.utils.data import DataLoader

datareader = Datareader("ua.base", size = 0)
train = TrainDataset(datareader.train, datareader.user_df, datareader.items_df)
test = TestDataset(datareader.test, datareader.user_df, datareader.items_df)

train_loader = DataLoader(train)
test_loader = DataLoader(test)

# urw = UnweightedRandomWalk(train)
net = EmbeddingNetwork()

for i, (v, lbl) in enumerate(train_loader):

    out = net(v)

for i, (v, name, cats, id) in enumerate(test_loader):
    out = net(v)