import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool, AttentionalAggregation, \
    JumpingKnowledge

from torch.nn import Linear
from utils.utils import pair_norm



class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, aggr_method, pool='max'):
        super().__init__()
        self.emb_in = Linear(2, hidden_channels)
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[pool]
        self.conv1 = GCNConv(hidden_channels, hidden_channels, aggr=aggr_method)
        self.conv2 = GCNConv(hidden_channels, hidden_channels, aggr=aggr_method)
        self.conv3 = GCNConv(hidden_channels, hidden_channels, aggr=aggr_method)
        self.pair_norm = pair_norm()
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.pos, data.edge_index, data.edge_attr, data.batch
        x = self.emb_in(x).relu()

        h1 = self.conv1(x, edge_index, edge_attr).relu()
        h1 = self.pair_norm(h1)

        h2 = self.conv2(h1, edge_index, edge_attr).relu()
        h2 = self.pair_norm(h2)

        h3 = self.conv3(h2, edge_index, edge_attr).relu()
        h3 = self.pair_norm(h3)

        h = self.pool(h3, batch)

        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return h
