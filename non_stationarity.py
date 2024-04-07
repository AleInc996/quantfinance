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
The unit root is found when the p value is high and the t-statistic produces a low value. More specifically, when the p value is
higher than the level of significance (given that the test is built with 0.01, 0.05 and 0.1 levels, we go for the medium one, 0.05).
When the opposite happens, then we will reject the null hypothesis in favor of the alternative one and
we will select the last tried model.

The possible model are:
    
    1. "n": without drift and without trend
    2. "c": with drift and without trend
    3. "ct": with drift and with trend
    4. "ctt": with drift, with trend and assuming linear and quadratic relationship
"""

model_list = ["n", "c", "ct", "ctt"]
for i in model_list:
    x = ADF(stock_prices, trend = i, method = "bic") # we can alternatively select aic method, instead of bic
    if x.pvalue >= 0.05:
        print("With model selection ", i, " the process has a unit root and it is not weakly stationary. Proceeding with next model selection")
        continue
    else:
        print("The process is weakly stationary, due to the fact that the p-value ", x.pvalue, " is less than significance level 0.05")
        print("The model selected is ", i)
        break

"""
In the default example for Apple prices, we observe that we never reject the null hypothesis.

One way to confirm this is to run another statistical test on the selected model or on the more complicated model: the KPSS one.
The KPSS aims at checking, as well, the presence of stationarity in a time series.

The difference with the ADF test is that the reasoning on the p-value is opposite: accepting the null hypothesis means that there is no unit root,
and that therefore the process is weakly stationary.
Therefore, please make sure that the model selected produces a low p-value under KPSS test.
"""

kpss = KPSS(stock_prices, trend = "ct") # KPSS test for time series with drift and trend, ctt is not available here
if kpss.pvalue >= 0.05:
    print("The process is weakly stationary, due to the fact that the p-value ", kpss.pvalue, " is higher than significance level 0.05")
else:
    print("The process is not weakly stationary, due to the fact that the p-value ", kpss.pvalue, " is less than significance level 0.05")

"""
Here we observed that no model selection allows us to have the stock prices time series in a stationary setting.
Therefore, we move on with considering returns instead of prices.
"""

#-------------------------------------------------------------------------------------------------

### Replicating analysis with returns instead of prices, because usually main analyses involve returns rather than prices

stock_returns = stock_prices.pct_change(1) # calculating daily returns
stock_returns = stock_returns.dropna() # dropping first observation because no return is produced

plt.plot(stock_returns) # plotting stock returns to eventually visualize non-stationarity
plt.xlabel("Time")
plt.ylabel(ticker_choice)
plt.title("Plot of daily stock returns")
plt.show()

for i in model_list:
    x = ADF(stock_returns, trend = i, method = "bic") # we can alternatively select aic method, instead of bic
    if x.pvalue >= 0.05:
        print("With model selection ", i, " the process has a unit root and it is not weakly stationary. Proceeding with next model selection")
        continue
    else:
        print("The process is weakly stationary, due to the fact that the p-value ", x.pvalue, " is less than significance level 0.05")
        print("The model selected is ", i)
        break

"""
In the default example for Apple, we observe that we reject null hypothesis on tentative 1.
This means that the time series considered does not have a drift and trend.

This time, instead of running the KPSS test for confirmation, we run Variance Ratio test, Phillips-Perron test and Zivot Andrews test.
The logic behind this three, in terms of approach to assess the resulted p-value and t-statistic, is the opposite than for KPSS.
"""
# Variance Ratio test
var_ratio = VarianceRatio(stock_returns, 12) # 1-period return is compared with multiperiod return, we selected 12-period, where each period is a month
if var_ratio.pvalue >= 0.05:
    print("The process is not weakly stationary, due to the fact that the p-value ", var_ratio.pvalue, " is higher than significance level 0.05")
else:
    print("The process is weakly stationary, due to the fact that the p-value ", var_ratio.pvalue, " is less than significance level 0.05")


# Phillips-Perron test
phil_perr = PhillipsPerron(stock_returns, trend = 'n') # selecting model 'n' because in default example for Apple returns this is the identified model
if phil_perr.pvalue >= 0.05:
    print("The process is not weakly stationary, due to the fact that the p-value ", phil_perr.pvalue, " is higher than significance level 0.05")
else:
    print("The process is weakly stationary, due to the fact that the p-value ", phil_perr.pvalue, " is less than significance level 0.05")


# Zivot Andrews test
ziv_and = ZivotAndrews(stock_returns, method='bic')
if ziv_and.pvalue >= 0.05:
    print("The process is not weakly stationary, due to the fact that the p-value ", ziv_and.pvalue, " is higher than significance level 0.05")
else:
    print("The process is weakly stationary, due to the fact that the p-value ", ziv_and.pvalue, " is less than significance level 0.05")


"""
But your question would be: why all of this? Why checking for stationarity or non-stationarity of a time series is so important?

"""