import sys

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch
from dataloader import preprocess_data
from model import STTF
import argparse
from  utils import metric


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="data_sub", help="the datasets name")
parser.add_argument('--mRNA_num',type=int,default="4880",help="the number of mRNA")
parser.add_argument('--miRNA_num',type=int,default="70",help="the number of miRNA")
parser.add_argument('--heads', type=int, default=4, help="The number of heads of multi-head attention")
parser.add_argument('--dropout', type=float, default=0, help="Dropout")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--embed_size', type=float, default=64, help="Embed_size")
parser.add_argument('--epochs', type=int, default=100, help="epochs")
parser.add_argument('--device',type = str,default="cpu")
parser.add_argument('--batchsize', type=int, default=4, help="Number of training batches")




def run(args):
    Data_X,Data_Y =  preprocess_data(args.mRNA_num)


    train_X = Data_X[0:1100,:]
    train_Y = Data_Y[0:1100,:]

    test_X = Data_X[1100:,:]
    test_Y = Data_Y[1100:,:]


    mean,std = np.mean(train_X),np.std(train_X)



    train_X = (train_X-mean)/std
    test_X = (test_X-mean)/std


    train_X = torch.tensor(train_X,dtype=torch.float32)
    train_Y = torch.tensor(train_Y,dtype=torch.float32)

    test_X = torch.tensor(test_X,dtype=torch.float32)
    test_Y = torch.tensor(test_Y,dtype=torch.float32)

    adj = torch.tensor(np.load("data_sub/adj.npy"),dtype=torch.float32)

    train_data = TensorDataset(train_X,train_Y)
    train_dataloader = DataLoader(train_data,batch_size = args.batchsize,shuffle=True)

    test_data = TensorDataset(test_X,test_Y)
    test_dataloader = DataLoader(test_data,batch_size = args.batchsize)

    device = args.device
    criterion = nn.MSELoss()
    model = STTF(adj,args.mRNA_num+args.miRNA_num,args.miRNA_num,1,args.embed_size,args.heads,4,args.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in tqdm(range(args.epochs)):
        for x,y in train_dataloader:
            model.train()
            x = x.to(device)
            y = y.to(device)
            pre = model(x)


            loss = criterion(pre*std+mean,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            P = []
            L = []
            for x,y in test_dataloader:
                x = x.to(device)
                pre = model(x)*std+mean
                P.append(pre.cpu().detach())
                L.append(y)

            pre = torch.cat(P,0)
            label = torch.cat(L,0)

            mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
            print("rmse,mae,mape,wape:", rmse, mae, mape, wape)


if __name__ == "__main__":
    args = parser.parse_args()
    run(args)
