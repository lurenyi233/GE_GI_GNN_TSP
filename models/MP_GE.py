import torch
from torch.nn import Linear, ReLU, SiLU, Sequential
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
import torch.nn.functional as F
from utils.utils import pair_norm
from torch_cluster import knn_graph

# reference: https://github.com/chaitjo/geometric-gnn-dojo/blob/main/models/layers/egnn_layer.py
class MP_GE_Layer(MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]
        self.mlp_msg = Sequential(Linear(2 * emb_dim + 1, emb_dim), self.activation)
        self.mlp_pos = Sequential(Linear(emb_dim, 2), self.activation)
        self.mlp_upd = Sequential(Linear(2 * emb_dim, emb_dim), self.activation)

    def forward(self, h, pos, edge_index):
        out = self.propagate(edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        pos_diff = pos_diff * self.mlp_pos(msg)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos

class MP_GE(torch.nn.Module):
    def __init__(self, num_layers=2, emb_dim=16, in_dim=2, out_dim=2, activation="relu", norm="layer", aggr="sum", pool="max", residual=False, dynamic_edge="DE", positional_encoding='RW'):
        super().__init__()
        self.emb_in = Linear(5, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MP_GE_Layer(emb_dim, activation, norm, aggr))
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[pool]
        self.lin1 = Linear(emb_dim, emb_dim)
        self.lin2 = Linear(emb_dim, 2)
        self.residual = residual
        self.positional_encoding = positional_encoding
        self.dynamic_edge = dynamic_edge

    def forward(self, data):
        if self.positional_encoding is None:
            data.h = torch.zeros(data.pos.size()[0], 5)
        h = self.emb_in(data.h).relu()
        pos = data.pos
        edge_index = data.edge_index
        for conv in self.convs:
            h_update, pos_update = conv(h, pos, edge_index)
            h = h + h_update if self.residual else h_update
            h = pair_norm()(h)
            pos = pos_update
            if self.dynamic_edge is not None:
                edge_index = knn_graph(pos, k=10, batch=data.batch, loop=False)
        h = self.pool(h, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return h
