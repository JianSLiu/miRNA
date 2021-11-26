import numpy as np
import torch
import pandas as pd


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


def Get_rel_dic(mRNA_num,miRNA_num):
    rel = pd.read_csv("data_sub/Rel.csv")

    rel_dic = {}
    for index in range(mRNA_num, mRNA_num+miRNA_num):
        rel_dic[index] = []

    for i in rel.index:
        rel_dic[int(rel.iloc[i, 0].split("_")[1])].append(int(rel.iloc[i, 1].split("_")[1]))

    return rel_dic

print(Get_rel_dic(4880,70))