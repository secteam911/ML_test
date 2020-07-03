import yfinance as yf
import os
from pandas_datareader import data as pdr
from clint.textui import puts ,colored, indent
from alpha_vantage.timeseries import TimeSeries
import sys
import time
import random 
import h5py

## pandas for dataframes manipulation 
import pandas as pd 
from datetime import datetime 
import dateparser
from clint.textui import puts , indent , colored 
## for protecting API keys  
## importing binance for assets_info fetching
from binance.client import *
from binance.enums import * 


from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout



##  for working with decimal numbers 
from decimal import Decimal

## importing tensorflow 
import tensorflow as tf
from tensorflow import keras


import matplotlib.dates as mdates

import matplotlib.pyplot as plt 

import matplotlib.transforms as mtransforms 
import numpy as np 





import time 
import datetime



portfolio = ["BNB-USD"]#,"BTC-USD" , "AAPL" , "GOOGL" , "AMZN" , "GC=F" , "NVDA", "AMD", "NFLX" , "TSLA"]



def get_data(item):
    
    ts = TimeSeries(key='DTVWPKMI7NQ4L7AI', output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=item,interval='1min', outputsize='full')

    dff = data.head(240)

    return dff






def feed_me(df , item ):



    checkpoint_filepath = "E://py_stuff/BIN_API_v3/python-binance-master/y_finance"
    os.chdir(checkpoint_filepath)


    print("Number of questions: ", df.shape[0])
    dataset_train = df

    print (dataset_train.head(5))
    print (dataset_train.tail(5))


    training_set = dataset_train.iloc[:, 1:2].values
    print (training_set)

    print (type(training_set))
     



    print (training_set , dataset_train)



    dataset_train.head()
    print (len(dataset_train))



    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)


    X_train = []
    y_train = []
    for i in range(30, 240):
        X_train.append(training_set_scaled[i-30:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print (X_train)
    model = Sequential()
    model.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 100, activation='tanh', return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 100, activation='tanh',return_sequences = True))
    model.add(Dropout(0.2))
                                   # 
    model.add(LSTM(units = 100 , activation= "tanh"))
    model.add(Dropout(0.2))

    model.add(Dense(units = 100 , activation="relu"))
    model.add(Dropout(0.2))


    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error'  ,metrics=['accuracy'])



    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_acc',
        mode='max',
        save_best_only=True)




    try:
        tf.keras.models.load_model('model')
    except :
        print ("No weights found")   

     

   
        
    model.summary()

    model.fit(X_train, y_train, epochs = 100 , batch_size = 7 )

    model.save('model')



if __name__ == "__main__":
    while True:
        for item in portfolio:
            print (item)

            dff = get_data(item)

            feed_me(df=dff , item=item )
