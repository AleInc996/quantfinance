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

ticker_choice = 'AAPL'
start_date = '2010-01-01'
end_date = '2023-12-31'
prices = get_usable_data(ticker_choice, start_date, end_date)