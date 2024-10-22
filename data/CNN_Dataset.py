import tsplib95
import networkx as nx
import numpy as np
import pandas as pd

import torch
import os, os.path
from collections import defaultdict
from typing import List, Optional, Union
from torch_geometric.data import Dataset, Data


class TSPDataset(Dataset):

    def __init__(self, root, split, folds, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """
        self.split = split
        self.folds = folds
        self.dir = str(os.path.split(os.path.realpath(__file__))[0])
        self.csvname = os.path.join(self.dir, "tsp-instances from CNN/sophisticated_1000_folds.csv")
        super(TSPDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)
        """
        return []

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.csvname)
        keys = self.data['Path']
        self.data = self.data.set_index('Path')

        if self.split == 'train':
            return [f'{key}.pt' for key in list(keys) if self.data.loc[key]['Fold'] != self.folds]
        elif self.split == 'val':
            return [f'{key}.pt' for key in list(keys) if self.data.loc[key]['Fold'] == self.folds]

    def download(self):
        pass

    def from_networkx_complete(self, G, group_node_attrs: Optional[Union[List[str], all]] = None,
                               group_edge_attrs: Optional[Union[List[str], all]] = None):
        G = nx.convert_node_labels_to_integers(G)

        # remove self loop edges
        G.remove_edges_from(nx.selfloop_edges(G))

        # to be directed graphs
        G = G.to_directed() if not nx.is_directed(G) else G

        data = defaultdict(list)

        # get node feautes
        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        # 添加点的信息
        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            data['coord'].append(feat_dict['coord'])

        for key, value in data.items():
            try:
                data[key] = torch.tensor(np.array(value, dtype=float))
            except ValueError:
                pass

        if group_node_attrs is not None:
            xs = []
            for key in group_node_attrs:
                x = data[key]
                x = x.view(-1, 1) if x.dim() <= 1 else x
                xs.append(x)
            pyg_data = Data(pos=torch.cat(xs, dim=-1).float())

        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)

        return pyg_data

    def process(self):
        print('loading data...')

        self.data = pd.read_csv(self.csvname)
        keys = self.data['Path']
        self.data = self.data.set_index('Path')

        DIR = os.path.join(self.dir, "tsp-instances from CNN/TSP")

        for key in keys:
            tsp_dir = os.path.join(DIR, key)
            print(tsp_dir)
            tsp_instance = tsplib95.load(tsp_dir)
            tsp_nxgraph = tsp_instance.get_graph()
            tsp_pygdata = self.from_networkx_complete(tsp_nxgraph, group_node_attrs=['coord'],
                                                      group_edge_attrs=['weight'])
            tsp_pygdata.y = torch.argmin(
                torch.tensor([self.data.loc[key]['EAX.PAR10'], self.data.loc[key]['LKH.PAR10']]))
            if self.pre_transform is not None:
                tsp_pygdata = self.pre_transform(tsp_pygdata)
            path, file = os.path.split(os.path.join(self.processed_dir, f'{key}.pt'))
            os.makedirs(path, exist_ok=True)
            torch.save(tsp_pygdata, os.path.join(self.processed_dir, f'{key}.pt'))

    def len(self):
        if self.split == 'train':

            return 900
        else:
            return 100

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        self.data = pd.read_csv(self.csvname)
        keys = self.data['Path']
        self.data = self.data.set_index('Path')

        if self.split == 'train':
            train_list = [f'{key}.pt' for key in list(keys) if self.data.loc[key]['Fold'] != self.folds]
            data = torch.load(os.path.join(self.processed_dir, train_list[idx]))
        elif self.split == 'val':
            val_list = [f'{key}.pt' for key in list(keys) if self.data.loc[key]['Fold'] == self.folds]
            data = torch.load(os.path.join(self.processed_dir, val_list[idx]))
        return data


dataset_path = os.path.join(str(os.path.split(os.path.realpath(__file__))[0]), "TSPDataset/")
print(dataset_path)

dataset_train = TSPDataset(root=dataset_path, split="train", folds=2)
dataset_test = TSPDataset(root=dataset_path, split="val", folds=2)

print(dataset_train[0])
print(dataset_test[0])