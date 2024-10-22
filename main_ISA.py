
import argparse
import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from torchsummary import summary

from data.ISA_Dataset import ISA_Dataset_Point

from models.GINE_GE import GINE_GE
from models.GINE_GI import GINE_GI
from models.MP_GE import MP_GE
from models.MP_GI import MP_GI

from models.GCN import GCN
from models.GINE import GINE
from models.Point_Clouds import Edge_CNN, PointNet
from utils.utils import seed_torch, train, test

parser = argparse.ArgumentParser(description='GNN for TSP algorithm selection and hardness prediction')
parser.add_argument('--data', type=str, default='ISA', help='dataset comes from')
parser.add_argument('--dataset', type=str, default='CLK', help='the type of dataset')
parser.add_argument('--folds', default=10, type=int, help='Number of Fold to use')
parser.add_argument('--k_values', default=10, type=int, help='Number of K in KNN graph')
parser.add_argument('--pos_feature', default='no', type=str, help='whether use pos as node feature')

parser.add_argument('--model', default='GINE_GI', type=str, help='GINES model')
parser.add_argument('--aggr', default='sum', type=str, help='Aggregation method for GNN layer')
parser.add_argument('--pool', default='max', type=str, help='Pooling method for GNN layer')
parser.add_argument('--h_dim', default=16, type=int, help='Hidden layer dimension')
parser.add_argument('--seed', default=41, type=int, help='Random seed')
parser.add_argument('--dynamic_edge', default=None, type=str, help='Dynamic edge')
parser.add_argument('--positional_encoding', default='RW', type=str, help='Positional encoding')
parser.add_argument('--algorithm', default='CLK', type=str, help='Positional encoding')
args = parser.parse_args()







if __name__ == '__main__':

    # create output dir
    result_file = os.path.join('output', args.data, args.dataset)
    if not os.path.exists(result_file):
        os.makedirs(result_file)

    dataset = ISA_Dataset_Point(root=os.path.join('data', args.data), dataset=args.dataset,
                                pre_transform=T.NormalizeScale())

    if args.positional_encoding == 'RW':
        dataset.transform = T.Compose([T.KNNGraph(k=args.k_values), T.Distance(norm=False),
                                       T.AddRandomWalkPE(walk_length=5, attr_name='h')])
    else:
        dataset.transform = T.Compose([T.KNNGraph(k=args.k_values), T.Distance(norm=False)])


    training_time = []


    seed_torch(seed=args.seed)
    csv_name = f"{args.algorithm}_{args.model}_{args.k_values}_{args.h_dim}_{args.positional_encoding}_{args.dynamic_edge}_{args.seed}.csv"

    kfold = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    train_result_kfold = []
    test_result_kfold = []

    fold = 1


    for train_ix, test_ix in kfold.split(dataset, dataset.data.y):
        print("=========", fold, "=========")

        starttime = datetime.datetime.now()
        if args.model == 'GINE_GE':
            model = GINE_GE(num_layers=2,
                            emb_dim=args.h_dim,
                            aggr=args.aggr,
                            pool=args.pool,
                            residual=False,
                            dynamic_edge=args.dynamic_edge,
                            positional_encoding=args.positional_encoding)

        elif args.model == 'GINE_GI':
            model = GINE_GI(hidden_channels=args.h_dim, aggr_method=args.aggr,
                            positional_encoding=args.positional_encoding, pool=args.pool)

        elif args.model == 'MP_GE':
            model = MP_GE(num_layers=2,
                          emb_dim=args.h_dim,
                          activation="relu",
                          norm="layer",
                          aggr=args.aggr,
                          pool="max",
                          residual=False,
                          dynamic_edge=args.dynamic_edge,
                          positional_encoding=args.positional_encoding)
        elif args.model == 'MP_GI':
            model = MP_GI(num_layers=2,
                          emb_dim=args.h_dim,
                          activation="relu",
                          norm="layer",
                          aggr=args.aggr,
                          pool="max",
                          residual=False,
                          dynamic_edge=args.dynamic_edge,
                          positional_encoding=args.positional_encoding)

        elif args.model == 'GCN':
            model = GCN(hidden_channels=args.h_dim, aggr_method=args.aggr, pool=args.pool)
        elif args.model == 'GINE':
            model = GINE(hidden_channels=args.h_dim, aggr_method=args.aggr, pool=args.pool)

        elif args.model == 'PointNet':
            model = PointNet(hidden_channels=args.h_dim, k=10, pool=args.pool)
        elif args.model == 'Edge_CNN':
            model = Edge_CNN(hidden_channels=args.h_dim, k=10,  pool=args.pool)

        summary(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        train_loader = DataLoader(dataset[torch.LongTensor(train_ix)], batch_size=32, shuffle=True)
        test_loader = DataLoader(dataset[torch.LongTensor(test_ix)], batch_size=200, shuffle=False)

        train_acc_list = []
        test_acc_list = []

        epoch = 0
        train_acc = test(model, train_loader)
        train_acc_list.append(train_acc)
        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)

        print(f'Epoch: {epoch:02d}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        optimizer.zero_grad()
        for epoch in range(1, 151):
            loss = train(model, optimizer, train_loader, criterion)

            train_acc = test(model, train_loader)
            train_acc_list.append(train_acc)

            test_acc = test(model, test_loader)
            test_acc_list.append(test_acc)
            print \
                (f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')
        train_result_kfold.append(train_acc_list)
        test_result_kfold.append(test_acc_list)

        model_name = f"{args.algorithm}_{args.model}_{args.k_values}_{args.h_dim}_{args.positional_encoding}_{args.dynamic_edge}_{args.seed}_{fold}.pth"
        torch.save(model.state_dict(), result_file + model_name)

        fold = fold + 1

    train_csv_name = f"Train_{args.algorithm}_{args.model}_{args.k_values}_{args.h_dim}_{args.positional_encoding}_{args.dynamic_edge}_{args.seed}.csv"
    test_csv_name = f"Test_{args.algorithm}_{args.model}_{args.k_values}_{args.h_dim}_{args.positional_encoding}_{args.dynamic_edge}_{args.seed}.csv"


    pd.DataFrame(np.array(train_result_kfold, ndmin=2)).to_csv(result_file + train_csv_name, mode='a', index=True, header=False)
    pd.DataFrame(np.array(test_result_kfold, ndmin=2)).to_csv(result_file + test_csv_name, mode='a', index=True, header=False)
