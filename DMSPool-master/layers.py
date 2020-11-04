from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch.nn import Parameter
import torch
import torch.nn.functional as F


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh, nhid=32):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.nhid = nhid
        # score_layer用于计算自注意力得分，这个得分既考虑了节点的结构信息，又考虑了节点特征信息；
        ###################################################################
        # DMSPool模型的原本的多架构----3路
        self.score_layer_first = Conv(in_channels, 1)

        self.score_layer_second = torch.nn.ModuleList()
        self.score_layer_second.append(Conv(in_channels, nhid))
        self.score_layer_second.append(Conv(nhid, 1))

        self.score_layer_third = torch.nn.ModuleList()
        self.score_layer_third.append(Conv(in_channels, nhid))
        self.score_layer_third.append(Conv(nhid, nhid))
        self.score_layer_third.append(Conv(nhid, 1))
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        x_feature1 = x
        x_feature2 = x
        x_feature3 = x

        # DMSPool模型的单尺度(3)多架构
        x_feature1 = F.relu(self.score_layer_third[0](x_feature1, edge_index))
        x_feature1 = F.relu(self.score_layer_third[1](x_feature1, edge_index))
        scoreP1 = self.score_layer_third[2](x_feature1, edge_index).squeeze(-1)

        x_feature2 = F.relu(self.score_layer_third[0](x_feature2, edge_index))
        x_feature2 = F.relu(self.score_layer_third[1](x_feature2, edge_index))
        scoreP2 = self.score_layer_third[2](x_feature2, edge_index).squeeze(-1)

        x_feature3 = F.relu(self.score_layer_third[0](x_feature3, edge_index))
        x_feature3 = F.relu(self.score_layer_third[1](x_feature3, edge_index))
        scoreP3 = self.score_layer_third[2](x_feature3, edge_index).squeeze(-1)
        ############################################################################
        score = scoreP1 + scoreP2 + scoreP3

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(scoreP1[perm]).view(-1, 1)

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm