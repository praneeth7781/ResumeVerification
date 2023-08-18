import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import torch
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import numpy as np
import pickle as pkl




companies = ['GOOG','AMZN','TSLA','NVDA','DELL','AAPL', 'GME', 'IBM', 'MSFT']



if(__name__ == '__main__'):

    for comp in companies:
        df = yf.download(comp,start = "2007-01-01")
        df.to_csv('./DATA/{}_2007.csv'.format(comp))

        df['Close'].plot()
        plt.title(comp)
        plt.ylabel('Closing Price')
        plt.xlabel('Days')
        plt.savefig('./DATA/{}_2007.png'.format(comp))
        plt.show()


    for stock in companies:
        df = pd.read_csv('./DATA/{}_2007.csv'.format(stock))
        corr = df.corr()
        sn.heatmap(corr,annot=True)
        plt.show()

