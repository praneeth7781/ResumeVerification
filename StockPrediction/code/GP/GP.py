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

from utils import MinMaxScale2, make_dataset



#############################################
#<--------------HYPERPARAMS---------------->#
WINDOW = 64
PRED = 1
INPUT = 1




EPOCHS = 10
############################################



def kernel(X,Y,mu,sigma):
    """
    X : (n,1)
    Y : (m,1)

    return K(.,.) : (n,m)
    """

    Cross_sub = torch.tensor(X - Y.T)

    return mu*torch.exp(-Cross_sub**2/(2*sigma**2))





if( __name__ == "__main__"):


    stock = sys.argv[1]
    year = sys.argv[2]

    name = "{}_{}".format(stock,year)

    #-----------------------------DATASET------------------|
    df = pd.read_csv('./DATA/{}.csv'.format(name))
    data = np.array(df['Close'])

    data, smin, smax = MinMaxScale2(data)

    X_train,Y_train = make_dataset(data,INPUT,WINDOW,PRED)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,shuffle=True,train_size=0.8)



    #--------------------------------------------------------------------------------[][][]


    mu = torch.tensor(1.0,requires_grad=True)
    sigma = torch.tensor(1.0,requires_grad=True)
    noise = torch.tensor(0.5,requires_grad=True)
    


    optim = torch.optim.SGD([mu,sigma],lr = 0.01)


    for epoch in range(EPOCHS):
        N,m,_ = X_train.shape
        x = np.arange(m).reshape(-1,1)

        K = kernel(x,x,mu,sigma)
        C = K + torch.eye(m)*(noise**2)

        y_avg = torch.mean()

            #Negative log Marginal LIkelyhood
            objective = y.T@torch.inverse(C)@y - torch.log(torch.det(C))


            print(objective)


            optim.zero_grad()
            objective.backward()
            optim.step()

        print("epoch {}/{}".format(epoch,EPOCHS))
        print(mu,sigma,noise)


            










