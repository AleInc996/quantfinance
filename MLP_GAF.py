# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:58:26 2025

@author: AleInc996
"""

"""

"""

# importing necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import sys
import yfinance.shared as shared
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# function for data retrieval first defined in non_stationarity code of the repo
def get_usable_data(ticker_choice, start_date, end_date):
    
    "This function simply accesses yahoo finance API to obtain prices data and returns an error if the written ticker does not exist"
    
    x = yf.download(ticker_choice, start = start_date, end = end_date, multi_level_index = False, auto_adjust = False) # returns daily prices of the selected ticker
    if not list(shared._ERRORS.keys()):
        print(x)
    else:
        print('The ticker', ticker_choice, 'does not exist or may be removed from Yahoo finance API, please use another one', file = sys.stderr)
    return x

ticker_choice = 'AAPL' # choosing the stock we want to analyze
start_date = '2015-01-01' # choosing beginning of the period
end_date = '2023-12-31' # choosing end of the period
stock_prices = get_usable_data(ticker_choice, start_date, end_date) # running the function

stock_prices["Returns_current"] = stock_prices["Adj Close"].pct_change() # calculating returns by taking adjusted close percentage changes of prices; we call it current as it will be the variable which will be explained by past returns
stock_rets = stock_prices.reset_index().dropna() # taking the dates column outside of the index, as we will need to work with it, and dropping the first obvious NA observation
stock_rets = stock_rets.drop(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'], axis = 1) # limiting the dataset to only returns, because that is what we need

# the idea is to make a sort of momentum study and analyze how current returns (the output variable) depend on returns from the past 21, 42, 63, 126 and 240 days.
# those number of days are chosen because they represent the trading days in one month, 2 months, 3 months and 6 months
# however feel free to change them as you wish

# defining variables for the rolling days as it is needed for columns names
roll1 = 21 # first variable rolling days
roll2 = 42 # second variable rolling days
roll3 = 63 # third variable rolling days
roll4 = 126 # fourth variable rolling days

# calculating past rolling returns
stock_rets['Returns_' + str(roll1)] = stock_rets['Returns_current'].rolling(roll1).apply(lambda x: 100 * (np.prod(1 + x / 100) - 1))
stock_rets['Returns_' + str(roll2)] = stock_rets['Returns_current'].rolling(roll2).apply(lambda x: 100 * (np.prod(1 + x / 100) - 1))
stock_rets['Returns_' + str(roll3)] = stock_rets['Returns_current'].rolling(roll3).apply(lambda x: 100 * (np.prod(1 + x / 100) - 1))
stock_rets['Returns_' + str(roll4)] = stock_rets['Returns_current'].rolling(roll4).apply(lambda x: 100 * (np.prod(1 + x / 100) - 1))

stock_rets = stock_rets.dropna() # dropping NAs again as we don't have values for all starting dates given the timeframes considered

stock_rets['Results'] = stock_rets['Returns_current'] >= 0 # for an easy interpretability, it is better to transform the current returns into a series of 0 for negative returns and 1 for non-strictly positive returns, so that classification of results will be easier
stock_rets['Results'] = stock_rets['Results'].astype(int) # turning the values of the results column into integers
stock_rets = stock_rets.drop('Returns_current', axis = 1) # we no longer need the proper columns with current returns, we will be using only the 0-1 column
stock_rets = stock_rets.reset_index(drop = True) # resetting index as we dropped NAs

X = stock_rets[['Returns_' + str(roll1), 'Returns_' + str(roll2), 'Returns_' + str(roll3), 'Returns_' + str(roll4)]] # defining X matrix, that will be used for training and testing
y = stock_rets['Results'] # defining y matrix, that will be used for training and testing
X = X.astype("float32") # making sure the X matrix is float, as that is needed for the train_test_split function
y = y.astype("float32") # making sure the y matrix is float, as that is needed for the train_test_split function

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int(0.3 * len(stock_rets)), shuffle = False)
# shuffle equal to False means that the dataset won't be shuffled before the split, the test set is established at 30% but it can be modified by changing the 0.3
# this means that the training dataset will correspond to 70% of the whole dataset

## model development part
tf.keras.backend.clear_session()  # clearing the session to make sure that the random seed initialization starts now
tf.random.set_seed(42)  # random seed is usually set to get always the same results (you can choose the number to insert)

activation_function = "relu" # relu is the activation function selected
dim_units = 25 # dimensionality of output space for first layer
dim2_units = 15 # dimensionality of output space for second layer
dim3_units = 10 # dimensionality of output space for third layer
dropout_rate = 0.2 # dropout rate

model = tf.keras.models.Sequential() # the sequential method allows us to stack together multiple layers of the deep learning model
model.add(tf.keras.layers.Dense(dim_units, activation_function)) # first layer, which takes as inputs dimensionality of output space and activation function
model.add(tf.keras.layers.Dropout(dropout_rate)) # dropout rate of the first layer
model.add(tf.keras.layers.Dense(dim2_units, activation_function)) # second layer
model.add(tf.keras.layers.Dropout(dropout_rate)) # dropout rate of the second layer
model.add(tf.keras.layers.Dense(dim3_units, activation_function)) # third layer
model.add(tf.keras.layers.Dropout(dropout_rate)) # dropout rate of the third layer
model.add(tf.keras.layers.Dense(units = 1, activation = "sigmoid")) # output layer, with dimensionality space equal to 1 and sigmoid activation function, as we want outputs to be binary

learning_rate = 1e-5  # defining the learning rate
adam = tf.keras.optimizers.Adam(learning_rate)  # Adam is the optimizer selected (feel free to use something else), which takes the learning rate as input

model.compile(optimizer = adam, loss = "binary_crossentropy", metrics = ["accuracy"])

## model training
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    patience=20,
    restore_best_weights=True,
)


class_weight = {0: (np.mean(y_train) / 0.5) * 1.2, 1: 1.0}
print(class_weight)


history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    verbose=2,
    callbacks=[es],
    class_weight=class_weight,
)

model.summary()
