import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from dataloader import preprocess_data
from model6 import STTF
import argparse
from  utils import metric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data_sub", help="the datasets name")
parser.add_argument('--mRNA_num', type=int, default="4880", help="the number of mRNA")
parser.add_argument('--miRNA_num', type=int, default="70", help="the number of miRNA")
parser.add_argument('--heads', type=int, default=1, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.0001, help="Learning rate")
parser.add_argument('--embed_size', type=float, default=2, help="Embed_size")
parser.add_argument('--epochs', type=int, default=200, help="epochs")
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--batchsize', type=int, default=4, help="Number of training batches")
parser.add_argument('--train_sampile', type=int, default=10000, help="Number of training samples")


args = parser.parse_args()
Data_X,Data_Y =  preprocess_data(args.mRNA_num)

train_X = Data_X[0:args.train_sampile, :]
train_Y = Data_Y[0:args.train_sampile, :]

test_X = Data_X[args.train_sampile:, :]
test_Y = Data_Y[args.train_sampile:, :]

mean,std = np.mean(train_X),np.std(train_X)

train_X = (train_X - mean) / std
test_X = (test_X - mean) / std

adj = torch.tensor(np.load("data_sub/adj.npy"),dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y = torch.tensor(test_Y, dtype=torch.float32)

test_data = TensorDataset(test_X, test_Y)
test_dataloader = DataLoader(test_data, batch_size=args.batchsize)

device = args.device
model = STTF(adj,args.mRNA_num+args.miRNA_num,args.miRNA_num,1,args.embed_size,args.heads,4,args.dropout)
model.load_state_dict(torch.load("Model/2021-11-26/epoch+26+time 12-05-05.pkl"))
model = model.to(device)
model.eval()
with torch.no_grad():
    P = []
    L = []
    for x, y in test_dataloader:
        x = x.to(device)
        pre = model(x) * std + mean
        P.append(pre.cpu().detach())
        L.append(y)

    pre = torch.cat(P, 0)
    label = torch.cat(L, 0)

    mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
    print("rmse,mae,mape,wape:", rmse, mae, mape, wape)

