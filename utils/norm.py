import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/VITA-Group/Deep_GCN_Benchmarking/blob/main/tricks/tricks/norms.py
class mean_norm(torch.nn.Module):
    def __init__(self):
        super(mean_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        return x

class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class node_norm(torch.nn.Module):
    def __init__(self, node_norm_type="n", unbiased=False, eps=1e-5, power_root=2, **kwargs):
        super(node_norm, self).__init__()
        self.unbiased = unbiased
        self.eps = eps
        self.node_norm_type = node_norm_type
        self.power = 1 / power_root
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        # in GCN+Cora,
        # n v srv pr
        # 16 layer:  _19.8_  15.7 17.4 17.3
        # 32 layer:  20.3 _25.5_ 16.2 16.3

        if self.node_norm_type == "n":
            mean = torch.mean(x, dim=1, keepdim=True)
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = (x - mean) / std
        elif self.node_norm_type == "v":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / std

        elif self.node_norm_type == "m":
            mean = torch.mean(x, dim=1, keepdim=True)
            x = x - mean
        elif self.node_norm_type == "srv":  # squre root of variance
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.sqrt(std)
        elif self.node_norm_type == "pr":
            std = (
                    torch.var(x, unbiased=self.unbiased, dim=1, keepdim=True) + self.eps
            ).sqrt()
            x = x / torch.pow(std, self.power)
        return x

    def __repr__(self):
        original_str = super().__repr__()
        components = list(original_str)
        node_norm_type_str = f"node_norm_type={self.node_norm_type}"
        components.insert(-1, node_norm_type_str)
        new_str = "".join(components)
        return new_str


class group_norm(torch.nn.Module):
    def __init__(self, dim_to_norm=None, dim_hidden=16, num_groups=5, skip_weight=0.001, **w):
        super(group_norm, self).__init__()
        self.num_groups = num_groups
        self.skip_weight = skip_weight

        dim_hidden = dim_hidden if dim_to_norm is None else dim_to_norm
        self.dim_hidden = dim_hidden

        # print(f'\n\n{dim_to_norm}\n\n');raise

        self.bn = torch.nn.BatchNorm1d(dim_hidden * self.num_groups, momentum=0.3)
        self.group_func = torch.nn.Linear(dim_hidden, self.num_groups, bias=True)
        # print(f'------ ›››››››››  {self._get_name()}')

    def forward(self, x):
        if self.num_groups == 1:
            x_temp = self.bn(x)
        else:
            score_cluster = F.softmax(self.group_func(x), dim=1)
            x_temp = torch.cat([score_cluster[:, group].unsqueeze(dim=1) * x for group in range(self.num_groups)],
                               dim=1)
            x_temp = self.bn(x_temp).view(-1, self.num_groups, self.dim_hidden).sum(dim=1)

        x = x + x_temp * self.skip_weight
        return x
