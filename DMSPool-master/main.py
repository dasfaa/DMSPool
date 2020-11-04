import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
import numpy as np
from tensorboardX import SummaryWriter
import time
import shutil
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=512,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.0,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='PROTEINS',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--epochs', type=int, default=10000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=150,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity')
parser.add_argument('--aspect', type=int, default=3)
parser.add_argument('--multiblock', type=int, default=3,
                    help='multi-block attention')
parser.add_argument('--gamma', type=float, default=0.5)
args = parser.parse_args()

result = "./result.txt"
f = open(result, 'a+')
f.write(str(args))
f.write("\n")
f.close()

args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
model = Net(args).to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def test(model, loader):
    # 把实例化的model指定train/eval，eval()时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out, score1, score2, score3 = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_dep = (loss_dependence(score1, score2) +
                    loss_dependence(score2, score3) +
                    loss_dependence(score3, score2)) / 3

        loss += F.nll_loss(out, data.y, reduction='sum').item() + args.gamma * loss_dep
    return correct / len(loader.dataset), loss / len(loader.dataset)


def loss_dependence(emb1, emb2):
    dim = emb1.size(0)
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC


min_loss = 1e10
patience = 0
path = os.path.join('run', str(args.dataset), str(args.seed))
writer1 = SummaryWriter(path + '/train')
writer2 = SummaryWriter(path + '/val')
t = time.time()

for epoch in range(args.epochs):
    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(args.device)
        out, score1, score2, score3 = model(data)
        loss_class = F.nll_loss(out, data.y)

        loss_dep = (loss_dependence(score1, score2) +
                    loss_dependence(score2, score3,) +
                    loss_dependence(score3, score2)) / 3
        loss = loss_class + args.gamma * loss_dep
        print("Training loss:{}".format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    val_acc, val_loss = test(model, val_loader)
    writer1.add_scalar('loss_train', val_loss, epoch)
    writer2.add_scalar('loss_train', loss, epoch)
    print("Validation loss:{}\taccuracy:{}".format(val_loss, val_acc))
    if val_loss < min_loss:
        torch.save(model.state_dict(), 'latest.pth')
        print("Model saved at epoch{}".format(epoch))
        min_loss = val_loss
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break
writer1.close()
writer2.close()
model = Net(args).to(args.device)
model.load_state_dict(torch.load('latest.pth'))
test_acc, test_loss = test(model, test_loader)
print("Test accuarcy:{}".format(test_acc))
result = "./result.txt"
f = open(result, 'a+')
f.write(str(test_acc))
f.write("\n")
f.close()
