# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 22:06:19 2025

@author: AleInc996
"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import sys
import yfinance.shared as shared
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


# function for data retrieval first defined in non_stationarity code of the repo
def get_usable_data(ticker_choice, start_date, end_date):
    
    """
    This function simply accesses yahoo finance API to obtain prices data 
    and returns an error if the written ticker does not exist.
    In addition to the ticker, this function takes beginning and end dates as inputs.
    """
    
    x = yf.download(ticker_choice, start = start_date, end = end_date, multi_level_index = False, auto_adjust = False) # returns daily prices of the selected ticker
    if not list(shared._ERRORS.keys()):
        print(x)
    else:
        print('The ticker', ticker_choice, 'does not exist or may be removed from Yahoo finance API, please use another one', file = sys.stderr)
    return x

ticker_choice = 'AAPL' # choosing the stock we want to analyze
start_date = '2019-01-01' # choosing beginning of the period
end_date = '2022-12-31' # choosing end of the period
stock_prices = get_usable_data(ticker_choice, start_date, end_date) # running the function

stock_prices['Stock returns'] = stock_prices['Adj Close'].pct_change() # calculating returns as prices percentage changes
stock_prices = stock_prices[1:] # removing the first row of the whole dataframe as the first return is NaN
stock_prices = stock_prices.dropna() # dropping other NAs, if any, in the dataframe

stock_prices.reset_index(inplace = True) # resetting the index, just to make sure it is aligned
stock_data = stock_prices[['Date', 'Stock returns']] # keeping only necessary columns, as our analysis will be focused on returns

# plotting returns of the stock we selected
plt.figure(figsize = (12, 6)) # initialization of the plot and setting size of the plot
plt.plot(stock_data['Date'], stock_data['Stock returns'], label = f"{ticker_choice} stock returns", color = 'blue', marker = 'o') # plotting stock returns
plt.title(f"{ticker_choice} stock Returns (2019-2022)") # title of the plot
plt.xlabel("Time") # dates are on the x-axis
plt.ylabel("Stock returns") # stock returns are on the y-axis
plt.xticks(rotation = 90) # formatting the x-axis to show dates vertically
plt.grid(True) # grid on
plt.legend() # including legend in the plot
plt.show() # displaying the plot in the plots environment

# the same time interval is plotted only with shaded regions to display regimes (either bull or bear market, everything else is treated as a stagnant period)
plt.axvspan('2019-01-01', '2019-12-31', color = 'green', alpha = 0.2, label = 'Bull market') # the period right before the outbreak of Covid can be considered as a bull period
plt.axvspan('2020-02-19', '2020-03-23', color = 'red', alpha = 0.2, label = 'Bear market') # the very first month of the Covid outbreak markets went down dramatically, entering a temporary bear trend
plt.axvspan('2020-03-24', '2022-02-23', color = 'green', alpha = 0.2) # after Covid and before Russia-Ukraine conflict, probably there was something that can be considered as a bull market
plt.axvspan('2022-02-24', '2022-07-31', color = 'red', alpha = 0.2) # clear bear trend due to Russia-Ukraine conflict
plt.xticks(rotation = 90) # formatting the x-axis to show dates vertically
plt.legend()# including legend in the plot
plt.show() # displaying the plot in the plots environment

stock_data['50-day moving average'] = stock_data['Stock returns'].rolling(window = 50).mean() # computing stock returns 50-day moving average
stock_data['200-day moving average'] = stock_data['Stock returns'].rolling(window = 200).mean() # computing stock returns 200-day moving average

# plotting the stock returns and moving averages
plt.figure(figsize = (12, 6)) # initialization of the plot and setting size of the plot
plt.plot(stock_data['Date'], stock_data['Stock returns'], label = f"{ticker_choice} stock returns", color = 'blue') # plotting stock returns
plt.plot(stock_data['Date'], stock_data['50-day moving average'], label = "50-day Moving average", color = 'orange') # plotting stock returns 50-day moving average
plt.plot(stock_data['Date'], stock_data['200-day moving average'], label = "200-day Moving average", color = 'brown') # plotting stock returns 200-day moving average
plt.title(f"{ticker_choice} stock returns with 50 and 200 day moving averages (2019-2022)") # title of the plot
plt.xlabel("Date") # dates are on the x-axis
plt.ylabel("Stock returns") # stock returns are on the y-axis
plt.xticks(rotation = 90) # formatting the x-axis to show dates vertically
plt.grid(True) # grid on
plt.legend() # including legend in the plot
plt.show() # displaying the plot in the plots environment


## Markov autoregression model estimation with multiple possibilities for number of states
number_states = [2, 3, 4, 5]  # estimation under different numbers of states
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True)
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True)

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True  # enabling regime switching for AR coefficient
    )

    model_results_study1 = model.fit() # fitting the model
    
    
    # plotting the model results
    fig, axes = plt.subplots(2, figsize = (12, 6)) # initialization of the plot and setting size of the plot

    for i in range(number_state): # looping over number of states considered
        axes[0].plot(aligned_dates_sp, model_results_study1.smoothed_marginal_probabilities[i], label = f'State {i}') # plotting smoothed marginal probabilities with dates; smoothed refers to an estimate of the probability at time t using all data in the sample 
    axes[0].set_title(f'Markov regime-switching model with {number_state} states') # title of first subplot
    axes[0].set_ylabel('State probability') # state probabilities are on the y-axis of first subplot
    axes[0].legend() # including legend of the plot
    axes[0].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    axes[1].plot(aligned_dates_mp, model_results_study1.predict(), label = 'Forecasted stock returns') # plotting model predicted values of stock returns with dates
    axes[1].set_ylabel('Stock returns') # stock returns are on the y-axis of second subplot
    axes[1].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    plt.tight_layout() # tightening the layout
    plt.show() # displaying the subplots
    
    
## Markov autoregression model estimation with single possibility for number of states and allowing for constant or varying mean and/or variance
number_states = [2]  # estimation under only one number of states
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True)
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True)

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True,  # enabling regime switching for AR coefficient
        switching_trend = True # enabling regime switching for all trend coefficients
    )

    model_results_study2 = model.fit() # fitting the model
    
    
    # plotting the model results
    fig, axes = plt.subplots(2, figsize = (12, 6)) # initialization of the plot and setting size of the plot

    for i in range(number_state): # looping over number of states considered
        axes[0].plot(aligned_dates_sp, model_results_study2.smoothed_marginal_probabilities[i], label = f'State {i}') # plotting smoothed marginal probabilities with dates; smoothed refers to an estimate of the probability at time t using all data in the sample 
    axes[0].set_title(f'Markov regime-switching model with {number_state} states') # title of first subplot
    axes[0].set_ylabel('State probability') # state probabilities are on the y-axis of first subplot
    axes[0].legend() # including legend of the plot
    axes[0].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    axes[1].plot(aligned_dates_mp, model_results_study2.predict(), label = 'Forecasted stock returns') # plotting model predicted values of stock returns with dates
    axes[1].set_ylabel('Stock returns') # stock returns are on the y-axis of second subplot
    axes[1].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    plt.tight_layout() # tightening the layout
    plt.show() # displaying the subplots
    
    aic_study2 = model_results_study2.aic # calculating Akaike Information Criteria of model results
    bic_study2 = model_results_study2.bic # calculating Bayesian Information Criteria of model results

    print("AIC:", aic_study2) # printing AIC value
    print("BIC:", aic_study2) # printing BIC value


number_states = [2]  # estimation under only one number of states
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True)
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True)


for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True,  # enabling regime switching for AR coefficient
        switching_variance = True #
    )

    model_results_study3 = model.fit() # fitting the model
    
    
    # plotting the model results
    fig, axes = plt.subplots(2, figsize = (12, 6)) # initialization of the plot and setting size of the plot

    for i in range(number_state): # looping over number of states considered
        axes[0].plot(aligned_dates_sp, model_results_study3.smoothed_marginal_probabilities[i], label = f'State {i}') # plotting smoothed marginal probabilities with dates; smoothed refers to an estimate of the probability at time t using all data in the sample 
    axes[0].set_title(f'Markov regime-switching model with {number_state} states') # title of first subplot
    axes[0].set_ylabel('State probability') # state probabilities are on the y-axis of first subplot
    axes[0].legend() # including legend of the plot
    axes[0].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    axes[1].plot(aligned_dates_mp, model_results_study3.predict(), label = 'Forecasted stock returns') # plotting model predicted values of stock returns with dates
    axes[1].set_ylabel('Stock returns') # stock returns are on the y-axis of second subplot
    axes[1].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    plt.tight_layout() # tightening the layout
    plt.show() # displaying the subplots
    
    # Calculate AIC, BIC
    aic_study3 = model_results_study3.aic # calculating Akaike Information Criteria of model results
    bic_study3 = model_results_study3.bic # calculating Bayesian Information Criteria of model results

    print("AIC:", aic_study3) # printing AIC value
    print("BIC:", bic_study3) # printing BIC value
    

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True,  # enabling regime switching for AR coefficient
        switching_trend = True, # enabling regime switching for all trend coefficients
        switching_variance = True 
    )

    model_results_study4 = model.fit() # fitting the model
    
    
    # plotting the model results
    fig, axes = plt.subplots(2, figsize = (12, 6)) # initialization of the plot and setting size of the plot

    for i in range(number_state): # looping over number of states considered
        axes[0].plot(aligned_dates_sp, model_results_study4.smoothed_marginal_probabilities[i], label = f'State {i}') # plotting smoothed marginal probabilities with dates; smoothed refers to an estimate of the probability at time t using all data in the sample 
    axes[0].set_title(f'Markov regime-switching model with {number_state} states') # title of first subplot
    axes[0].set_ylabel('State probability') # state probabilities are on the y-axis of first subplot
    axes[0].legend() # including legend of the plot
    axes[0].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    axes[1].plot(aligned_dates_mp, model_results_study4.predict(), label = 'Forecasted stock returns') # plotting model predicted values of stock returns with dates
    axes[1].set_ylabel('Stock returns') # stock returns are on the y-axis of second subplot
    axes[1].tick_params(axis = 'x', rotation = 90) # formatting the x-axis to show dates vertically

    plt.tight_layout() # tightening the layout
    plt.show() # displaying the subplots
    
    aic_study4 = model_results_study4.aic # calculating Akaike Information Criteria of model results
    bic_study4 = model_results_study4.bic # calculating Bayesian Information Criteria of model results

    print("AIC:", aic_study4) # printing AIC value
    print("BIC:", bic_study4) # printing BIC value



# Example AIC and BIC values for different models
aic_values = [aic_study2, aic_study3, aic_study4]
bic_values = [bic_study2, bic_study3, bic_study4]

min_aic_index = np.argmin(aic_values) # looking for the position of the minimum AIC
min_bic_index = np.argmin(bic_values) # looking for the position of the minimum BIC

# Print the results (TODO: find a way to print name of the model of what we ran)
print("Choosing the model with smallest AIC:")
for i, aic in enumerate(aic_values):
    print(f"Model {i+1}: AIC = {aic} {'(Best)' if i == min_aic_index else ''}")

print("Choosing the model with smallest BIC:")
for i, bic in enumerate(bic_values):
    print(f"Model {i+1}: BIC = {bic} {'(Best)' if i == min_bic_index else ''}")
    
    
# displaying smoothed probabilities of low and high variance regimes for the type of model that we want to choose
# TODO: create an interface to select the model you want to display this for
fig, axes = plt.subplots(2, figsize=(10,7))
ax = axes[0]
ax.plot(results_all.smoothed_marginal_probabilities[0])
ax.grid(False)
ax.set(title='Smoothed probability of a low-variance regime returns')
ax = axes[1]
ax.plot(results_all.smoothed_marginal_probabilities[1])
ax.set(title='Smoothed probability of a high-variance regime returns')
fig.tight_layout()
ax.grid(False)
