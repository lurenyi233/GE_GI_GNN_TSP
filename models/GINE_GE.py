import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
from torch_geometric.nn.dense.linear import Linear
from torch.nn import ReLU, Sequential
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter
from torch_cluster import knn_graph
from utils.utils import pair_norm
from torch_geometric.nn import aggr


class GINE_GE_Layer(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.lin = Sequential(Linear(1, emb_dim), ReLU())
        self.mlp_msg = Sequential(Linear(emb_dim, emb_dim), ReLU())
        self.mlp_pos = Sequential(Linear(emb_dim, 2), ReLU())
        self.mlp_upd = Sequential(Linear(emb_dim, emb_dim), ReLU())

    def forward(self, h, pos, edge_index):
        out = self.propagate(edge_index=edge_index, h=h, pos=pos)
        return out

    def message(self, h_i, h_j, pos_i, pos_j) -> Tensor:
        pos_diff = pos_i - pos_j
        dists = torch.norm(pos_diff, dim=-1).unsqueeze(1)
        dists = self.lin(dists)
        msg = h_j + dists
        msg = self.mlp_msg(msg)
        pos_diff = pos_diff * self.mlp_pos(msg)
        return msg, pos_diff

    def aggregate(self, inputs, index):
        msgs, pos_diffs = inputs
        if self.aggr == 'std':
            std_aggr = aggr.StdAggregation()
            msg_aggr = std_aggr(msgs, index, dim=self.node_dim)
        elif self.aggr == 'var':
            var_aggr = aggr.VarAggregation()
            msg_aggr = var_aggr(msgs, index, dim=self.node_dim)
        else:
            msg_aggr = scatter(msgs, index, dim=self.node_dim, reduce=self.aggr)
        pos_aggr = scatter(pos_diffs, index, dim=self.node_dim, reduce="mean")
        return msg_aggr, pos_aggr

    def update(self, aggr_out, h, pos):
        msg_aggr, pos_aggr = aggr_out
        upd_out = h + msg_aggr
        upd_out = self.mlp_upd(upd_out)
        upd_pos = pos + pos_aggr
        return upd_out, upd_pos


class GINE_GE(torch.nn.Module):
    def __init__(self,
                 num_layers=2,
                 emb_dim=16,
                 aggr="sum",
                 pool="max",
                 residual=False,
                 dynamic_edge="DE",
                 positional_encoding='RW'):
        super().__init__()
        self.emb_in = Linear(5, emb_dim)
        self.convs = torch.nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(GINE_GE_Layer(emb_dim, aggr))
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
