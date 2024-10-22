import torch
from torch.nn import Linear, ReLU, SiLU, Sequential
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_scatter import scatter
import torch.nn.functional as F
from utils.utils import pair_norm
from torch_geometric.nn import aggr
from torch_cluster import knn_graph

class MP_GI_Layer(MessagePassing):
    def __init__(self, emb_dim, activation="relu", norm="layer", aggr="add"):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.activation = {"swish": SiLU(), "relu": ReLU()}[activation]
        self.norm = {"layer": torch.nn.LayerNorm, "batch": torch.nn.BatchNorm1d}[norm]
        self.mlp_msg = Sequential(Linear(2 * emb_dim + 1, emb_dim), self.activation)
        self.mlp_upd = Sequential(Linear(2 * emb_dim, emb_dim), self.activation)

    def forward(self, h, pos, edge_index):
        out = self.propagate(edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j):
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        msg = torch.cat([h_i, h_j, dists], dim=-1)
        msg = self.mlp_msg(msg)
        return msg

    def aggregate(self, inputs, index):
        msgs = inputs
        if self.aggr == 'std':
            std_aggr = aggr.StdAggregation()
            msg_aggr = std_aggr(msgs, index, dim=self.node_dim)
        elif self.aggr == 'var':
            var_aggr = aggr.VarAggregation()
            msg_aggr = var_aggr(msgs, index, dim=self.node_dim)
        else:
            msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        return msg_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr = aggr_out
        upd_out = self.mlp_upd(torch.cat([h, msg_aggr], dim=-1))
        return upd_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})"

class MP_GI(torch.nn.Module):
    def __init__(self, num_layers=2, emb_dim=16, activation="relu", norm="layer", aggr="sum", pool="max", residual=False, dynamic_edge="DE", positional_encoding='RW'):
        super().__init__()
        self.emb_in = Linear(5, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(MP_GI_Layer(emb_dim, activation, norm, aggr))
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
        for conv in self.convs:
            h_update = conv(h, pos, data.edge_index)
            h = h + h_update if self.residual else h_update
            h = pair_norm()(h)
            if self.dynamic_edge is not None:
                edge_index = knn_graph(pos, k=10, batch=data.batch, loop=False)
        h = self.pool(h, data.batch)
        h = self.lin1(h).relu()
        h = F.dropout(h, p=0.2, training=self.training)
        h = self.lin2(h)
        return h
