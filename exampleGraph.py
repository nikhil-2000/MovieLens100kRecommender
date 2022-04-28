from Datasets.Training import TrainDataset
from datareader import Datareader
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

users = ["A","B","C","D"]
movies = [1,2,3,4]

d = """A 1 3
A 3 2
B 4 5
C 2 1
C 3 4
D 4 5"""
d = [line.split(" ") for line in d.split("\n")]
print(d)

G.add_nodes_from(users, bipartite = 0)
G.add_nodes_from(movies, bipartite = 1)
edge_labels = {}
for u,v,w in d:
    v = int(v)
    e = (u, v)
    w = int(w)
    G.add_edge(u, int(v) ,weight = w)
    edge_labels[(u,v)] = str(w)


pos = nx.bipartite_layout(G, users)
nx.draw(G, pos, with_labels = True)
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_color='red',
    label_pos = 0.2
)
plt.show()
plt.savefig("bipartite_example.png")