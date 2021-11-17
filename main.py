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

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        wape = np.divide(np.sum(mae), np.sum(label))
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape, wape

#Node = 14152

Data_X,Data_Y =  preprocess_data()

#######################
#Data_X = Data_X[:,Node:]
#######################

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
#adj = adj[Node:,Node:]
train_data = TensorDataset(train_X,train_Y)
train_dataloader = DataLoader(train_data,batch_size = 1,shuffle=True)

test_data = TensorDataset(test_X,test_Y)
test_dataloader = DataLoader(test_data,batch_size = 1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
#adj,
#in_channels,
#embed_size,
#heads,
#forward_expansion,
#dropout=0
criterion = nn.MSELoss()
model = STTF(adj,4950,70,1,64,4,4,0)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in tqdm(range(100)):
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
            # print(y)
            # print("#"*50)
            # print(pre.cpu().detach())
            P.append(pre.cpu().detach())
            L.append(y)

        pre = torch.cat(P,0)
        label = torch.cat(L,0)

        mae, rmse, mape, wape = metric(pre.numpy(), label.numpy())
        print("rmse,mae,mape,wape:", rmse, mae, mape, wape)



