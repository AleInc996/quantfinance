# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 08:05:30 2024

@author: AleInc996

Often time, stock prices display a volatile behavior over time, with lots of ups and downs in price movements.
This means that, most probably, the mean of the observed prices won't stay the same, let alone the fact that
the magnitued of volatility can be quite different over some time intervals (see volatility clustering for more details on this).
However, volatility is not the only factor playing an important role in non-stationarity:
seasonality, trends and structural breaks in the input dataset are key points to be discussed.

This script analyzes the mentioned problem, leaving the user the freedom to select the desired data.
"""

### Stock modeling in the presence of non-stationarity

# importing main libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch.unitroot import ADF, KPSS
from arch.unitroot import PhillipsPerron
from arch.unitroot import ZivotAndrews
import yfinance as yf
import yfinance.shared as shared
import sys

# function for data retrieval
def get_usable_data(ticker_choice, start_date, end_date):
    
    "This function simply accesses yahoo finance API to obtain prices data and returns an error if the written ticker does not exist"
    
    x = yf.download(ticker_choice, start = start_date, end = end_date) # returns daily prices of the selected ticker
    if not list(shared._ERRORS.keys()):
        print(x)
    else:
        print('The ticker', ticker_choice, 'does not exist or may be removed from Yahoo finance API, please use another one', file = sys.stderr)
    return x

ticker_choice = 'AAPL' # choosing the stock we want to analyze
start_date = '2010-01-01' # choosing beginning of the period
end_date = '2023-12-31' # choosing end of the period
stock_prices = get_usable_data(ticker_choice, start_date, end_date) # running the function

stock_prices = stock_prices.dropna() # dropping NAs (rows with dividends)
stock_prices = stock_prices['Adj Close'] # taking adjusted close prices

plt.plot(stock_prices) # plotting stock prices to eventually visualize non-stationarity
plt.xlabel("Time")
plt.ylabel(ticker_choice)
plt.title("Plot of daily stock prices")
plt.show()

"""
In the Apple example, and probably in many other cases when it comes to stocks,
by looking at the time series plot we can already observe that the time series does not seem to be centered around zero and
does not seem to revert to a specific mean. Meaning, this time series probably displays a drift.
"""

# Computationally inspecting for time series characteristic
