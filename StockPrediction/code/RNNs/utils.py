import torch
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import sys
import time


#scale to (0,1)
def MinMaxScale(data):
    min_ = np.min(data)
    max_ = np.max(data)

    #data = (data - min_)/(max_ - min_) - 0.5
    data = data/max_

    return data, min_, max_


#SCales to range (-1,1)
def MinMaxScale2(data):
    min_ = np.min(data)
    max_ = np.max(data)

    data = 2*(data - min_)/(max_ - min_) - 1

    return data, min_, max_

def make_dataset(data,input,window,pred):

    data = data.reshape(-1,input)
    N,_ = data.shape
    X_train = np.zeros((N-window,window,input))
    Y_train = np.zeros((N-window,pred,input))
    for i in range(N-window-pred):
        X_train[i] = data[i:i+window,:]
        Y_train[i] = data[i+window:i+window+pred,:]


    return torch.tensor(X_train).to(torch.float32),torch.tensor(Y_train).to(torch.float32)

