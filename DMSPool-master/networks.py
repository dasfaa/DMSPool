import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from layers import SAGPool
from torch.nn.parameter import Parameter


class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.pooling_ratio = args.pooling_ratio
        self.dropout_ratio = args.dropout_ratio
        self.aspect = args.aspect
        self.max_k = args.multiblock

        self.block1, self.block2, self.block3 = self.build_block(self.aspect, self.num_features, self.nhid)

        block_merge_first_dim = 0
        for i in range(self.aspect):
            block_merge_first_dim = block_merge_first_dim + self.nhid * (i + 2)
        self.pool = SAGPool(block_merge_first_dim, ratio=self.pooling_ratio)

        self.block21, self.block22, self.block23 = self.build_block(self.aspect, block_merge_first_dim, self.nhid)

        # Node level attention
        self.k_weight = Parameter(torch.Tensor(block_merge_first_dim * args.multiblock, block_merge_first_dim))
        self.bn3 = torch.nn.BatchNorm1d(block_merge_first_dim)

        self.lin1 = torch.nn.Linear(block_merge_first_dim, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid//2)
        self.lin3 = torch.nn.Linear(self.nhid//2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_all = []
        x_tensor_list = self.massage_propagate(x, edge_index, self.block1, self.block2, self.block3)
        x_tensor = torch.cat(x_tensor_list, 1)
        x, edge_index, _, batch, _ = self.pool(x_tensor, edge_index, None, batch)  # x维度： [8281, 60]
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)       # x1维度：[64, 120]
        x1 = gmp(x, batch)
        x_all.append(x1)

        x_tensor_list2 = self.massage_propagate(x, edge_index, self.block21, self.block22, self.block23)
        x_tensor2 = torch.cat(x_tensor_list2, 1)
        x, edge_index, _, batch, _ = self.pool(x_tensor2, edge_index, None, batch)  # x维度：[4157, 60]
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)       # x2维度:[64,120]
        x2 = gmp(x, batch)
        x_all.append(x2)

        x_tensor_list3 = self.massage_propagate(x, edge_index, self.block21, self.block22, self.block23)
        x_tensor3 = torch.cat(x_tensor_list3, 1)
        x, edge_index, _, batch, _ = self.pool(x_tensor3, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)       # [64,120]
        x3 = gmp(x, batch)
        x_all.append(x3)

        n, m = x3.shape
        result = torch.zeros((n, m * self.max_k)).cuda()
        step = 0
        while step < self.max_k:
            result[:, (step * m):((step + 1) * m)] = x_all[step]
            step += 1
        # x = x1 + x2 + x3
        ######################################################
        # 注意力
        x = torch.matmul(result, self.k_weight).view(n, -1)
        # 平均
        # x = result.view((n, 3, m)).mean(1)
        # 最大
        # x = result.view((n, 3, m)).max(1)[0]
        # 最小
        # x = result.view((n, 3, m)).min(1)[0]
        #######################################################
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_ratio, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        return x, x1, x2, x3

    def build_block(self, num_aspect, input_dim, nhid):
        block_first = torch.nn.ModuleList()
        block_second = torch.nn.ModuleList()
        block_third = torch.nn.ModuleList()

        for i in range(num_aspect):
            block_first_temp = GCNConv(input_dim, nhid)
            block_first.append(block_first_temp)

            in_dim = nhid
            out_dim = nhid
            block_second_temp = torch.nn.ModuleList([GCNConv(in_dim, out_dim) for num in range(i)])
            block_second.append(block_second_temp)

            in_dim = nhid
            out_dim = nhid
            block_third.append(GCNConv(in_dim, out_dim))

        return block_first, block_second, block_third

    def massage_propagate(self, x, edge_index, block1, block2, block3):
        xlist = []
        edgelist = []
        length = len(block1)
        for i in range(length):
            xlist.append(x)
            edgelist.append(edge_index)
        x_tensor = []
        for i in range(length):
            x = xlist[i]
            edge = edgelist[i]
            x = F.relu(block1[i](x, edge))
            x_all = [x]
            temp_block2 = block2[i]
            for j in range(len(temp_block2)):
                x = F.relu(temp_block2[j](x, edge))
                x_all.append(x)
            x = F.relu(block3[i](x, edge))
            x_all.append(x)
            temp_x_tensor = torch.cat(x_all, dim=1)
            x_tensor.append(temp_x_tensor)
        return x_tensor

