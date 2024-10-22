import torch
import torch.nn.functional as F
from torch_geometric.nn import PointNetConv, DynamicEdgeConv
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_cluster import knn_graph
from torch.nn import Linear, Sequential
from utils.utils import pair_norm

class PointNet(torch.nn.Module):
    def __init__(self, hidden_channels, k=10, pool='max'):
        super().__init__()
        self.k = k
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[pool]
        self.pair_norm = pair_norm()

        self.conv1 = PointNetConv(local_nn=Linear(2, hidden_channels))
        self.conv2 = PointNetConv(local_nn=Linear(hidden_channels+2, hidden_channels))
        self.conv3 = PointNetConv(local_nn=Linear(hidden_channels+2, hidden_channels))

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, data):
        edge_index = knn_graph(data.pos, k=self.k, batch=data.batch, loop=False)

        h1 = self.conv1(None, data.pos, edge_index).relu()
        h1 = self.pair_norm(h1)

        h2 = self.conv2(h1, data.pos, edge_index).relu()
        h2 = self.pair_norm(h2)

        h3 = self.conv3(h2, data.pos, edge_index).relu()
        h3 = self.pair_norm(h3)

        h = self.pool(h3, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return h


class Edge_CNN(torch.nn.Module):
    def __init__(self, hidden_channels, k=10, pool='max'):
        super().__init__()
        self.k = k
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[pool]
        self.pair_norm = pair_norm()

        self.conv1 = DynamicEdgeConv(nn=Sequential(Linear(4, hidden_channels)), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(nn=Sequential(Linear(hidden_channels * 2, hidden_channels * 2)), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(nn=Sequential(Linear(hidden_channels * 4, hidden_channels * 4)), k=self.k, aggr='max')

        self.lin1 = Linear(hidden_channels*4, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, data):
        h1 = self.conv1(data.pos, data.batch).relu()
        h1 = self.pair_norm(h1)

        h2 = self.conv2(h1, data.batch).relu()
        h2 = self.pair_norm(h2)

        h3 = self.conv3(h2, data.batch).relu()
        h3 = self.pair_norm(h3)

        h = self.pool(h3, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return h
