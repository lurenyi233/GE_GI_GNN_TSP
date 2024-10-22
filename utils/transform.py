from typing import Optional

from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.utils import to_networkx, to_undirected
import networkx as nx
import math


class Distance_Feature(BaseTransform):

    def __init__(self, norm: bool = True, max_value: Optional[float] = None,
                 cat: bool = True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        distance = distance_matrix(data.pos, data.pos)
        mean_value = torch.mean(torch.tensor(distance), dim=0).reshape([-1, 1])

        max_value, _ = torch.max(torch.tensor(distance), dim=0)
        max_value = max_value.reshape([-1, 1])

        min_value, _ = torch.min(torch.tensor(distance), dim=0)
        min_value = min_value.reshape([-1, 1])
        var_value = torch.var(torch.tensor(distance), dim=0).reshape([-1, 1])
        h = torch.cat((mean_value, max_value, min_value, var_value), dim=-1).to(torch.float32)

        data.node_feat = h

        return data

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class OneHotDegree(BaseTransform):

    def __init__(
            self,
            k : int = 10,
            in_degree: bool = False,
    ):
        self.in_degree = in_degree

    def __call__(self, data: Data) -> Data:
        idx, x = data.edge_index[1 if self.in_degree else 0], data.x
        deg = degree(idx, data.num_nodes, dtype=torch.long)/(self.k)
        data.degree = deg

        return data




class Distance2D(BaseTransform):
    r"""Saves the Euclidean distance of linked nodes in its edge attributes
    (functional name: :obj:`distance`).

    Args:
        norm (bool, optional): If set to :obj:`False`, the output will not be
            normalized to the interval :math:`[0, 1]`. (default: :obj:`True`)
        max_value (float, optional): If set and :obj:`norm=True`, normalization
            will be performed based on this value instead of the maximum value
            found in the data. (default: :obj:`None`)
        cat (bool, optional): If set to :obj:`False`, all existing edge
            attributes will be replaced. (default: :obj:`True`)
    """
    def __init__(self, norm: bool = True, max_value: Optional[float] = None,
                 cat: bool = True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        (row, col), pos, pseudo = data.edge_index, data.pos, data.edge_attr

        #         dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        dist = pos[col] - pos[row]
        if self.norm and dist.numel() > 0:
            dist = dist / (dist.max() if self.max is None else self.max)

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, dist.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = dist

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')

class MST_Edge_Index(BaseTransform):

    def __init__(self, norm: bool = True, max_value: Optional[float] = None,
                 cat: bool = True):
        self.norm = norm
        self.max = max_value
        self.cat = cat

    def __call__(self, data: Data) -> Data:
        nx_graph = to_networkx(data, node_attrs=['pos'],edge_attrs=['edge_attr'] ).to_undirected()
        mst_graph = nx.minimum_spanning_tree(nx_graph, weight='edge_attr')
        edges = list(mst_graph.to_directed().edges)
        data.mst_edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        (row, col), pos, pseudo = data.mst_edge_index, data.pos, data.edge_attr
        dist = torch.norm(pos[col] - pos[row], p=2, dim=-1).view(-1, 1)
        data.mst_edge_attr = dist
        return data



class Pos_Rotation(BaseTransform):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, data: Data) -> Data:
        rotation_matrix = torch.tensor([[math.cos(self.angle), -math.sin(self.angle)],
                                        [math.sin(self.angle), math.cos(self.angle)]], dtype=torch.float32)
        data.pos = torch.round(torch.matmul(data.pos, rotation_matrix), decimals=5)
        return data

