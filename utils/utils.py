import numpy as np
import random
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data
from torch_geometric.transforms import Distance

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_tsp_graph(node_size=100, k=10):
    pos = torch.randn(node_size, 2)
    edge_index = knn_graph(pos, k=k, loop=False)
    data = Data(pos=pos, edge_index=edge_index)
    data = Distance()(data)
    return data

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x

def train(model, optimizer, loader, criterion):
    model.train()
    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

@torch.no_grad()
def test(model, loader):
    model.eval()
    total_correct = 0
    for data in loader:
        logits = model(data)
        pred = logits.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())
    return total_correct / len(loader.dataset)


def Draw_graph(data):
    fig, ax = plt.subplots(figsize=(6, 6))

    edge_index = data.edge_index
    pos = data.pos

    for i in range(edge_index.size(1)):
        x_coords = [pos[edge_index[0, i]][0], pos[edge_index[1, i]][0]]
        y_coords = [pos[edge_index[0, i]][1], pos[edge_index[1, i]][1]]
        ax.plot(x_coords, y_coords, 'o-', color='cyan')

    ax.set_title('MST Graph', fontsize=30)
    plt.tight_layout()
    plt.savefig('MST_Graph.jpg', dpi=300)
    plt.show()
