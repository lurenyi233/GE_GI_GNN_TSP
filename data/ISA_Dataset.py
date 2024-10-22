import os
import os.path
from collections import defaultdict
from typing import List, Optional, Union

import networkx as nx
import numpy as np
import torch
import tsplib95
from torch_geometric.data import InMemoryDataset, Data


class ISA_Dataset_Point(InMemoryDataset):
    def __init__(self, root, dataset='CLK', transform=None, pre_transform=None):
        self.dataset = dataset

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], map_location=self.device)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['TSP_data.pt']

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.dataset, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.dataset)


    def download(self):
        pass

    def from_networkx_complete(self, G, group_node_attrs: Optional[Union[List[str], all]] = None,
                               group_edge_attrs: Optional[Union[List[str], all]] = None):
        G = nx.convert_node_labels_to_integers(G)

        G.remove_edges_from(nx.selfloop_edges(G))

        G = G.to_directed() if not nx.is_directed(G) else G
        data = defaultdict(list)

        if G.number_of_nodes() > 0:
            node_attrs = list(next(iter(G.nodes(data=True)))[-1].keys())
        else:
            node_attrs = {}

        for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
            data['coord'].append(feat_dict['coord'])

        for key, value in data.items():
            try:
                data[key] = torch.tensor(np.array(value, dtype=float))
            except ValueError:
                pass

        pyg_data = Data(pos=data['coord'].float())

        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)

        return pyg_data

    def process_tsp_dataset(self, dataset_list, label, dir_path, g_list):

        for tspdata_name in dataset_list:
            print("Load", tspdata_name)
            tspdata_dir = os.path.join(dir_path, tspdata_name)
            for filename in os.listdir(tspdata_dir):
                if filename.endswith('.tsp'):
                    tsp_instance = tsplib95.load(os.path.join(tspdata_dir, filename))
                    tsp_nxgraph = tsp_instance.get_graph()
                    tsp_pygdata = self.from_networkx_complete(
                        tsp_nxgraph,
                        group_node_attrs=['coord'],
                        group_edge_attrs=['weight']
                    )
                    tsp_pygdata.y = torch.tensor(label)
                    g_list.append(tsp_pygdata)

    def process(self):
        print(f'Loading {self.dataset} TSP instances...')
        g_list = []

        DIR = r'data/tsp-instances from ISA'

        if self.dataset == 'CLK':
            self.process_tsp_dataset(
                dataset_list=['CLKeasy', 'easyCLK-hardLKCC'],
                label=0,
                dir_path=DIR,
                g_list=g_list
            )
            self.process_tsp_dataset(
                dataset_list=['CLKhard', 'hardCLK-easyLKCC'],
                label=1,
                dir_path=DIR,
                g_list=g_list
            )

        elif self.dataset == 'LKCC':
            self.process_tsp_dataset(
                dataset_list=['LKCCeasy', 'hardCLK-easyLKCC'],
                label=0,
                dir_path=DIR,
                g_list=g_list
            )
            self.process_tsp_dataset(
                dataset_list=['LKCChard', 'easyCLK-hardLKCC'],
                label=1,
                dir_path=DIR,
                g_list=g_list
            )
        elif self.dataset == 'AS':
            self.process_tsp_dataset(
                dataset_list=['CLKeasy', 'LKCCeasy', 'CLKhard', 'random', 'hardCLK-easyLKCC'],
                label=0,
                dir_path=DIR,
                g_list=g_list
            )
            self.process_tsp_dataset(
                dataset_list=['LKCChard', 'easyCLK-hardLKCC'],
                label=1,
                dir_path=DIR,
                g_list=g_list
            )

        else:
            raise ValueError("Invalid TSP type. Choose between 'CLK', 'LKCC' and 'AS'.")

        data, slices = self.collate(g_list)
        torch.save((data, slices), self.processed_paths[0])

