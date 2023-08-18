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


from utils import MinMaxScale, make_dataset


#############################################
#<--------------HYPERPARAMS---------------->#
WINDOW = 64
PRED = 1         #number of days to predict
HIDDEN = 32     #hidden unit size.
INPUT = 1       #dimension of input = 1, since we only use Closing price.


EPOCHS = 300
LR = 0.005
BATCH = 256
############################################

class MYRNN(torch.nn.Module):
    def __init__(self, input = 1, hidden = 64, output = 16,  layers = 1) -> None:
        super(MYRNN,self).__init__()
        self.input_size = input
        self.hidden_size = hidden
        self.output_size = output       #output size at each time step.


        ####### PRE PROCESSING ######
        self.scalemax = None
        self.scalemin = None

        self.rnn = torch.nn.RNN(input_size = input,hidden_size = hidden,num_layers = layers,nonlinearity = 'relu',\
                                bias = True, batch_first = True, dropout = 0)
        self.fc = torch.nn.Linear(in_features = hidden,out_features=output,bias=True)

    def set_scale(self,smin,smax):
        self.scalemax = smax
        self.scalemin = smin

    def forward(self,x, hidden):
        """
            x is of shape -> (BATCH,WINDOW,INPUT)
            returns (out,h)
                out -> (BATCH,PRED)
                h   -> (BATCH,WINDOW,HIDDEN)
        """
        out, hidden = self.rnn(x,hidden)

        #we just care abt the output at last time step in the window.
        out = out[:,-1,:].reshape((-1,self.hidden_size))

        out = self.fc(out)

        return out.unsqueeze(2),hidden

def train(model,X_train,Y_train,X_val,Y_val,lrate,batch_size,epochs, name, supress = False):
    N,window,input = X_train.shape
    _,_,output = Y_train.shape

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)


    TRAINLOSSES = []
    VALLOSSES = []
    VALID_MIN = 9999999
    TRAINMIN = 9999999
    bestmodel = None

    start = time.time()
    for epoch in range(epochs):

        num_iter = N//batch_size
        i = 0
        
        Iter_loss = 0

        while( i < num_iter*batch_size ):
            loss = 0
            hidden = None

            #Forward.....
            X,Y = X_train[i:i+batch_size],Y_train[i:i+batch_size]
            out,hidden = model(X, hidden)

            #loss + Backprop.....
            loss = criterion(out , Y)
            Iter_loss += loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += batch_size

        Iter_loss /= num_iter*batch_size
        Iter_loss *= (model.scalemax**2)

        Ypred_val,_ = model(X_val, None)
        Val_loss = criterion(Ypred_val, Y_val)/(Y_val.shape[0])*(model.scalemax**2)

        TRAINLOSSES.append(Iter_loss.item())
        VALLOSSES.append(Val_loss.item())
        if(not supress):
            print("epoch: {}/{} :".format(epoch,epochs), "  Train = ",round(Iter_loss.item(),4), "Valid = ", round(Val_loss.item(),4) )


        if(Val_loss.item() < VALID_MIN):
            VALID_MIN = Val_loss.item()
            TRAINMIN = Iter_loss.item()
            bestmodel = model
            with open('./MODELS/RNN_{}.pkl'.format(name),'wb') as f:
                pkl.dump(model,f)

    end = time.time()
    print("=====================================================")
    print("Train Error: ", TRAINMIN)    
    print("Validation err: ",VALID_MIN)
    print("[ Training Time : {} seconds ]".format(end-start))
    print("=====================================================")
        
    plt.plot(range(5,len(TRAINLOSSES)),TRAINLOSSES[5:],label = 'Train MSE loss')
    plt.plot(range(5,len(VALLOSSES)),VALLOSSES[5:],label = 'Validation MSE loss')
    plt.xlabel('epochs')
    plt.legend()
    plt.show()



def predict(model_name,X_test):
    with open(model_name,'rb') as f:
        model = pkl.load(f)

    X_test = (X_test)/model.scalemax

    return model(X_test,None)[0]*model.scalemax


if(__name__ == '__main__'):

    stock = sys.argv[1]
    year = sys.argv[2]

    name = "{}_{}".format(stock,year)

    #-----------------------------DATASET------------------|
    df = pd.read_csv('./DATA/{}.csv'.format(name))
    data = np.array(df['Close'])

    data, smin, smax = MinMaxScale(data)

    X_train,Y_train = make_dataset(data,INPUT,WINDOW,PRED)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,shuffle=True,train_size=0.8)


    #uncomment to visualize data points.
    ########################################### VISUALISING THE DATA POINTS #####################
    # for i in range(10):
    #     x,y = X_val[i],Y_val[i]
    #     plt.plot(np.arange(x.shape[0]),x,label = 'input x [{} days]'.format(WINDOW))
    #     plt.plot(np.arange(x.shape[0],x.shape[0]+y.shape[0]),y, label = 'Target y [{} days]'.format(PRED))
    #     plt.legend()
    #     plt.xlabel('Consecutive Days')
    #     plt.show()



    model = MYRNN(INPUT, HIDDEN, PRED, 1)
    model.set_scale(smin,smax)


    train(model,X_train,Y_train, X_val, Y_val, LR, BATCH, EPOCHS, name)

