#!/usr/bin/env python

# -*- coding: utf-8 -*-

# author:Hualei YU time:2020/9/9 20:52
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os


# Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中，我们再使用DataLoader这个类来更加快捷的对数据进行操作。
# 当我们集成了一个Dataset类之后，我们需要重写len方法，该方法提供了dataset的大小；getitem方法，该方法支持从 0 到 len(self)的索引
class GraphSampler(Dataset):
    def __init__(self, G_list, features='default', normalize=True, assign_feat='default', max_num_nodes=0):
        super(GraphSampler, self).__init__()
        self.adj_all = []  # 存所有图的标准化后的邻接矩阵；
        self.len_all = []  # 存所有图的节点个数；
        self.feature_all = []  # 存所有图的节点特征；
        self.label_all = []  # 存所有图的标签；
        self.features = []  # 存所有图的图特征--（3个，节点个数，边个数，度数）
        self.assign_feat_all = []

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        pass

