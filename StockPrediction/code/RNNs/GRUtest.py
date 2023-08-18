import torch
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle as pkl
from sklearn.model_selection import train_test_split
import sys


from GRU import MYGRU, make_dataset, train, predict, INPUT, PRED, EPOCHS, HIDDEN, WINDOW
from data_loader import companies


def compute_loss_1day(model_stock,model_yr,test_stock,test_yrs = [2007,2022], micro = False):
    """
        Though the model preicts next P days.. we only take the first of it( just next day)
        while finding loss as well as plotting.


        test_yrs --> range of years like [2007,2010] etc.
    """

    df = yf.download(test_stock,start='{}'.format(test_yrs[0]),end = '{}'.format(test_yrs[1]))
    data = np.array(df['Close'])

    X_test, Y_test = make_dataset(data,INPUT,WINDOW,PRED)

    Ypred = predict('./MODELS/GRU_{}_{}.pkl'.format(model_stock,model_yr),X_test).detach()
    criterion = torch.nn.MSELoss(reduction='mean')
    loss = criterion(Ypred,Y_test)

    print("MSE loss on {}_[{},{}] = ".format(test_stock,test_yrs[0],test_yrs[1]),loss.item(), " [{}_{}] ".format(model_stock,model_yr))

    #plotting onl the just next day pred vs true values
    Xaxis = range(Y_test.shape[0])
    plt.plot(Xaxis,Y_test[:,0,:],color = 'b', label = 'True prices')
    plt.plot(Xaxis,Ypred[:,0,:], color = 'r', label = 'Predicted')
    plt.legend()
    plt.title('Predicted prices on {} using {} [{}|{}]'.format(test_stock,model_stock,WINDOW,PRED))
    plt.show()


    if(micro):
        #testing some random micro (WINDOW,PRED) datapoints..
        for i in range(10):
            i = np.random.randint(0,X_test.shape[0])
            x,y, ypred = X_test[i], Y_test[i], Ypred[i]

            X1 = range(x.shape[0])
            X2 = range(x.shape[0],x.shape[0]+y.shape[0])


            plt.plot(X1,x,color = 'b',label = 'Input [{} days]'.format(WINDOW))
            plt.plot(X2,y,'g-o',label = 'True prices [next {} days]'.format(PRED))
            plt.plot(X2,ypred, 'r-o', label = 'Predicted [next {} days]'.format(PRED))
            plt.title('Predicted prices on {} using {} [{} ? {} days]'.format(test_stock,model_stock,WINDOW,PRED))
            plt.legend()
            plt.show()


    return Ypred, loss


if(__name__ == '__main__'):


    TEST_ALL = False

    model_stock = sys.argv[1]
    model_yr = sys.argv[2]
    
    if(len(sys.argv) == 3):
        test_stock = model_stock
        test_yr = ['{}-1-1'.format(model_yr), '2022-11-20']
    elif(len(sys.argv) == 4):
        test_stock = sys.argv[3]
        test_yr = ['2007-1-1', '2022-11-20']
        if(test_stock == 'ALL'):
            TEST_ALL = True
    elif(len(sys.argv) == 5):
        test_stock = sys.argv[3]
        test_yr = [sys.argv[4],'2022-11-20']
        if(test_stock == 'ALL'):
            TEST_ALL = True
    elif(len(sys.argv) == 6):
        test_stock = sys.argv[3]
        test_yr = [sys.argv[4], sys.argv[5]]
        if(test_stock == 'ALL'):
            TEST_ALL = True


    #--------------------------------------------------------------


    if(TEST_ALL):
        for comp in companies:
            compute_loss_1day(model_stock,model_yr,comp,test_yr, False)

    else:
        compute_loss_1day(model_stock,model_yr,test_stock,test_yr, True)
