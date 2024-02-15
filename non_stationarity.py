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
import matplotlib.pyplot as plt
from arch.unitroot import ADF, KPSS
from arch.unitroot import VarianceRatio
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

### Computationally inspecting for time series characteristic
"""
Visually inspecting for non-stationarity is not enough.
One of the most used and common statistical approach to detect non-stationarity is the Augmented Dickey Fuller (ADF) test.

The idea is to try from the simplest model (no drift and no trend) 
to the most complicated one (drift plus trend plus linearity plus quadratic) until we no longer find a unit root in the test.
The unit root is found when the p value is high and the t-statistic produces a low value.
When the opposite happens, then we will reject the null hypothesis in favor of the alternative one and
we will select the last tried model.
"""

## Tentative 1: ADF test for time series without drift and without trend
adf_no_drift_no_trend = ADF(stock_prices, trend = "n", method = "bic") 
print("ADF Unit Root Test summary \n", adf_no_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_no_drift_no_trend) # printing p-value and t-statistic

adf_no_drift_no_trend = ADF(stock_prices, trend = "n", method = "aic")
print("ADF Unit Root Test summary \n", adf_no_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_no_drift_no_trend) # printing p-value and t-statistic

## Tentative 2: ADF test for time series with drift and without trend
adf_yes_drift_no_trend = ADF(stock_prices, trend = "c", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_no_trend) # printing p-value and t-statistic

adf_yes_drift_no_trend = ADF(stock_prices, trend = "c", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_no_trend) # printing p-value and t-statistic

## Tentative 3: ADF test for time series with drift and with trend
adf_yes_drift_yes_trend = ADF(stock_prices, trend = "ct", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend) # printing p-value and t-statistic

adf_yes_drift_yes_trend = ADF(stock_prices, trend = "ct", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend) # printing p-value and t-statistic

## Tentative 4: ADF test for time series with drift, with trend and assuming linear and quadratic relationship
adf_yes_drift_yes_trend_q = ADF(stock_prices, trend = "ctt", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend_q.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend_q) # printing p-value and t-statistic

adf_yes_drift_yes_trend_q = ADF(stock_prices, trend = "ctt", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend_q.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend_q) # printing p-value and t-statistic

"""
In the default example for Apple, we observe that we reject null hypothesis on tentative 4.
This means that the time series considered has a drift and a trend.

One way to confirm this is to run another statistical test on the selected model: the KPSS one.
The KPSS aims at checking, as well, the presence of stationarity in a time series.

The difference with the ADF test is that the reasoning on the p-value is opposite: accepting the null hypothesis means that there is no unit root.
Therefore, please make sure that the model selected produces a low p-value under KPSS test.
"""

KPSS(stock_prices, trend = "ct") # KPSS test for time series with drift and trend

#-------------------------------------------------------------------------------------------------

### Replicating analysis with returns instead of prices, because usually main analyses involve returns rather than prices

stock_returns = stock_prices.pct_change(1) # calculating daily returns
stock_returns = stock_returns.dropna() # dropping first observation because no return is produced

plt.plot(stock_returns) # plotting stock returns to eventually visualize non-stationarity
plt.xlabel("Time")
plt.ylabel(ticker_choice)
plt.title("Plot of daily stock returns")
plt.show()

## Tentative 1: ADF test for time series without drift and without trend
adf_no_drift_no_trend = ADF(stock_returns, trend = "n", method = "bic") 
print("ADF Unit Root Test summary \n", adf_no_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_no_drift_no_trend) # printing p-value and t-statistic

adf_no_drift_no_trend = ADF(stock_returns, trend = "n", method = "aic")
print("ADF Unit Root Test summary \n", adf_no_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_no_drift_no_trend) # printing p-value and t-statistic

## Tentative 2: ADF test for time series with drift and without trend
adf_yes_drift_no_trend = ADF(stock_returns, trend = "c", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_no_trend) # printing p-value and t-statistic

adf_yes_drift_no_trend = ADF(stock_returns, trend = "c", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_no_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_no_trend) # printing p-value and t-statistic

## Tentative 3: ADF test for time series with drift and with trend
adf_yes_drift_yes_trend = ADF(stock_returns, trend = "ct", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend) # printing p-value and t-statistic

adf_yes_drift_yes_trend = ADF(stock_returns, trend = "ct", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend) # printing p-value and t-statistic

## Tentative 4: ADF test for time series with drift, with trend and assuming linear and quadratic relationship
adf_yes_drift_yes_trend_q = ADF(stock_returns, trend = "ctt", method = "bic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend_q.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend_q) # printing p-value and t-statistic

adf_yes_drift_yes_trend_q = ADF(stock_returns, trend = "ctt", method = "aic")
print("ADF Unit Root Test summary \n", adf_yes_drift_yes_trend_q.regression.summary()) # printing regression summary
print("\nTest statistics and critical values: \n", adf_yes_drift_yes_trend_q) # printing p-value and t-statistic

"""
In the default example for Apple, we observe that we reject null hypothesis on tentative 1.
This means that the time series considered does not have a drift and trend.

One way to confirm this is to run other statistical tests on the selected model: the KPSS one.
The KPSS aims at checking, as well, the presence of stationarity in a time series.

This time, in addition to KPSS test, we also run Variance Ratio test, Phillips-Perron test and Zivot Andrews test.
The logic behind this three, in terms of approach to assess the resulted p-value and t-statistic, is the same as for KPSS.
"""
# Variance Ratio test
var_ratio = VarianceRatio(stock_returns, 12) # 1-period return is compared with multiperiod return, we selected 12-period, where each period is a month
var_ratio.summary() # checking results

# Phillips-Perron test
phil_perr = PhillipsPerron(stock_returns, trend = 'n') # selecting model 'n' because in default example for Apple returns this is the identified model
phil_perr.summary()

# Zivot Andrews test
ziv_and = ZivotAndrews(stock_returns, method='bic')
ziv_and.summary()

"""
But your question would be: why all of this? Why checking for stationarity or non-stationarity of a time series is so important?

"""