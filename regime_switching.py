# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 22:06:19 2025

@author: AleInc996
"""

"""

As it is commonly recognized in academia, financial time series display a non-stationary behavior.
Leaving aside whether this non-stationarity will be shown from a mean or a variance point of view,
the main consequence of this point lies in the fact that a financial time series will, very often and unexpectedly,
display trends, that can last for short or long time intervals, seasonality or abrupt increases or decreases in prices.
Moreover, a financial time series can show specific patterns for a certain time interval and then forget about them for some other time interval.

Of course, these impacts on financial time series might originate from the overall economic scenario and environment:
this is why regime switching modeling is so important. Traditional time series models imply that we define some model parameters
and they can be employed to describe how the whole data changes over time; however, this is most of the time not true,
as mean and variances usually vary across different time periods and different regimes of the market.

One of the economic theories out there says that we can identify the current financial market situation among three possible regimes:
    - bull regime: prices are trending upwards;
    - bear regime: prices are trending downwards;
    - neutral regime: stagnation.
    
Regime-switching modeling allows for the definition of these regimes and to let model parameters vary accordingly.
The scope of this script is to test and model an equity time series as a regime switching time series model, using a Markov Autoregression model, 
with data running through pre-COVID times in 2019, pandemic, and come up through the fourth quarter of 2022 (end of December 2022).

For a narrative type of run, please stop running the code immediately before some other text spaces like this one,
in order to understand step by step what is happening.

"""

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import yfinance.shared as shared
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression


# function to identify the type of model that was run
def model_identification():
    
    """
    This function identifies the type of Markov autoregression model that was selected
    in order to be printed with final analyses.
    """
    
    model_name = [] # pre-allocating an empty variable for the name of the model that will be identified
    
    # creating the name of type of model run
    if len(number_states) > 1: # if number of states under analysis is greater than 1
        if model.switching_ar == [True] and model.switching_trend == [False] and model.switching_variance == False:
            model_name = 'multiple_states_ARswitching'
        elif model.switching_ar == [False] and model.switching_trend == [True] and model.switching_variance == False:
            model_name = 'multiple_states_TRENDswitching'
        elif model.switching_ar == [False] and model.switching_trend == [False] and model.switching_variance == True:
            model_name = 'multiple_states_VARIANCEswitching'
        elif model.switching_ar == [True] and model.switching_trend == [True] and model.switching_variance == False:
            model_name = 'multiple_states__AR_TRENDswitching'
        elif model.switching_ar == [True] and model.switching_trend == [False] and model.switching_variance == True:
            model_name = 'multiple_states__AR_VARIANCEswitching'
        elif model.switching_ar == [False] and model.switching_trend == [True] and model.switching_variance == True:
            model_name = 'multiple_states__TREND_VARIANCEswitching'
        elif model.switching_ar == [True] and model.switching_trend == [True] and model.switching_variance == True:
            model_name = 'multiple_states__AR_TREND_VARIANCEswitching'
    elif len(number_states) == 1: # if only one number of states is under analysis
        if model.switching_ar == [True] and model.switching_trend == [False] and model.switching_variance == False:
            model_name = 'single_state_ARswitching'
        elif model.switching_ar == [False] and model.switching_trend == [True] and model.switching_variance == False:
            model_name = 'single_state_TRENDswitching'
        elif model.switching_ar == [False] and model.switching_trend == [False] and model.switching_variance == True:
            model_name = 'single_state_VARIANCEswitching'
        elif model.switching_ar == [True] and model.switching_trend == [True] and model.switching_variance == False:
            model_name = 'single_state__AR_TRENDswitching'
        elif model.switching_ar == [True] and model.switching_trend == [False] and model.switching_variance == True:
            model_name = 'single_state__AR_VARIANCEswitching'
        elif model.switching_ar == [False] and model.switching_trend == [True] and model.switching_variance == True:
            model_name = 'single_state__TREND_VARIANCEswitching'
        elif model.switching_ar == [True] and model.switching_trend == [True] and model.switching_variance == True:
            model_name = 'single_state__AR_TREND_VARIANCEswitching'
            
    return model_name
    
    
# function for data retrieval first defined in non_stationarity code of the repo
def get_usable_data(ticker_choice, start_date, end_date, frequency = '1d'):
    
    """
    This function simply accesses yahoo finance API to obtain prices data 
    and returns an error if the written ticker does not exist.
    In addition to the ticker, this function takes beginning and end dates as inputs.
    
    The need for this function became useful after recent changes to yahoo finance library in Python,
    therefore standardizing the use of this library.
    """
    
    x = yf.download(ticker_choice, start = start_date, end = end_date, interval = frequency, multi_level_index = False, auto_adjust = False) # returns daily prices of the selected ticker
    if not list(shared._ERRORS.keys()):
        print(x)
    else:
        print('The ticker', ticker_choice, 'does not exist or may be removed from Yahoo finance API, please use another one', file = sys.stderr)
    return x



ticker_choice = 'AAPL' # choosing the stock we want to analyze
start_date = '2019-01-01' # choosing beginning of the period
end_date = '2022-12-31' # choosing end of the period
frequency = '1d' # choosing the frequency of prices data, it can be one of the following: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
stock_prices = get_usable_data(ticker_choice, start_date, end_date) # running the function

stock_prices['Stock returns'] = stock_prices['Adj Close'].pct_change() # calculating returns as prices percentage changes
stock_prices = stock_prices[1:] # removing the first row of the whole dataframe as the first return is NaN
stock_prices = stock_prices.dropna() # dropping other NAs, if any, in the dataframe

stock_prices.reset_index(inplace = True) # resetting the index, just to make sure it is aligned
stock_data = stock_prices[['Date', 'Stock returns']] # keeping only necessary columns, as our analysis will be focused on returns

"""
By plotting the selected equity time series with the below plotting code block,
we will realize that for most of the stocks variance is never really constant: periods of high volatility are followed by
periods of lower volatility (volatility clustering).
"""

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

"""
With the following plotting code block, bull and bear regimes are defined.
Feel free to change them as you please, for the moment this is the way they are defined:
    
    - Jan 2019-Dec 2019: Bull regime
    - Feb 2020-Mar 2020: Bear regime (COVID crash)
    - Mar 2020-Feb 2022: Bull regime
    - Feb 2022-Jul 2022: Bear regime (Russia-Ukraine conflict)
    - Aug 2022-Dec 2022: Neutral regime
"""

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

# plotting the stock returns and moving averages, as sometimes short-term moving average crossing long-term moving average might be a bull signal (viceversa might signal the beginning of a bear trend)
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


models_name = [] # pre-allocating memory for a list where the name of the model identified will be stored
aic_values = [] # pre-allocating memory for AIC values of the models run
bic_values = [] # pre-allocating memory for BIC values of the models run

"""
Below follows the run of different versions of the Markov autoregression model.
The code is automated for you to change freely the number of states and switching_ar, switching_trend and switching_variance parts of the models.

The code runs 4 different versions of the Markov autoregression approach, and for each one of them plots the state probabilities and forecasted stock returns.
In addition to this, only for model with single states run, AIC and BIC values are found, and the name of the model run is saved for further comparison.

Once models are run, plots are visible in the Plots section of the environment on the right.
"""

## Markov autoregression model estimation with multiple possibilities for number of states
number_states = [2, 3, 4, 5]  # estimation under different numbers of states, you can change this to whatever you wish
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True) # aligning length of dates to length of smoothed marginal probabilities
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True) # aligning length of dates to length of model forecasted stock returns

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True, # enabling regime switching for AR coefficient
        switching_trend = False, # not enabling regime switching for trend coefficient
        switching_variance = False # not enabling regime switching for variance coefficient
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
    
#----------------------------------------------------------------------------------------------------------------
    
    
## Markov autoregression model estimation with single possibility for number of states and allowing for constant or varying mean and/or variance
number_states = [2]  # estimation under only one number of states, you can set this to whatever number you wish
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True) # aligning length of dates to length of smoothed marginal probabilities
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True) # aligning length of dates to length of model forecasted stock returns

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True, # enabling regime switching for AR coefficient
        switching_trend = True, # and for all trend coefficients
        switching_variance = False # not enabling regime switching for variance coefficient
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
    
model2_name = model_identification() # identifying model name by looking at its features
models_name.append(model2_name) # appending identified model name to the initially created variable
aic_values.append(aic_study2) # appending AIC value to the initially created variable
bic_values.append(bic_study2) # appending BIC value to the initially created variable


#----------------------------------------------------------------------------------------------------------------

number_states = [2]  # estimation under only one number of states, you can set this to whatever number you wish
aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True) # aligning length of dates to length of smoothed marginal probabilities
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True) # aligning length of dates to length of model forecasted stock returns


for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True, # enabling regime switching for AR coefficient
        switching_trend = False, # not for all trend coefficients
        switching_variance = True # and for
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
    
model3_name = model_identification() # identifying model name by looking at its features
models_name.append(model3_name) # appending identified model name to the initially created variable
aic_values.append(aic_study3) # appending AIC value to the initially created variable
bic_values.append(bic_study3) # appending BIC value to the initially created variable


#----------------------------------------------------------------------------------------------------------------

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = True, # enabling regime switching for AR coefficient
        switching_trend = True, # and for all trend coefficients
        switching_variance = True # and for variance coefficient
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
    
model4_name = model_identification() # identifying model name by looking at its features
models_name.append(model4_name) # appending identified model name to the initially created variable
aic_values.append(aic_study4) # appending AIC value to the initially created variable
bic_values.append(bic_study4) # appending BIC value to the initially created variable


"""
Among the 3 models for which we produced AIC and BIC values and for which the name was stored,
we might want to select the model which has the most conversative AIC or BIC.
"""

min_aic_pos = np.argmin(aic_values) # looking for the position of the minimum AIC
min_bic_pos = np.argmin(bic_values) # looking for the position of the minimum BIC


if len(models_name) != len(aic_values): # making sure that the number of models run is equal to the number of produced AIC values
    print("Number of models run should be the same as AIC values produced. Check that everything was run properly.")
else:
   print("Choosing the model with smallest AIC:")
   for i, aic in enumerate(aic_values):
       print(f"Model {models_name[i]}: AIC = {aic} {'(Best)' if i == min_aic_pos else ''}") 


if len(models_name) != len(bic_values): # making sure that the number of models run is equal to the number of produced BIC values
    print("Number of models run should be the same as BIC values produced. Check that everything was run properly.")
else:
    print("Choosing the model with smallest BIC:")
    for i, bic in enumerate(bic_values):
        print(f"Model {models_name[i]}: BIC = {bic} {'(Best)' if i == min_bic_pos else ''}")
        

"""
Being aware of what the last two blocks of code produced, in the console we will see the suggestion (Best) of 
which model we should select according to both AIC and BIC approach.
Run the following block of code for choosing the final model and the remaining part of the code will do the rest.
"""

## interactively choosing one model among the ones analyzed
choices = models_name # the possible choices are the models run during the analysis
model_selection = '' # pre-allocating memory for the choice that the user will make
input_message = "Which among the following models would you like to visualize smoothed probabilities of low and high variance regimes for?\n" # question

for index, model in enumerate(choices): # looping over possible choices
    input_message += f'{index+1}) {model}\n' # showing possible choices

input_message += 'Your choice: ' # message to ask for the choice
model_selection = input(input_message) # giving the user the possibility to write his/her own choice
print('You picked: ' + model_selection) # printing final choice




if 'single_state' in model_selection: # if the selected model has only one number of states
    number_states = [2]  # estimation under only one number of states, you can set this to whatever number you wish
elif 'multiple_states' in model_selection: # if the selected model has more than 1 number of states
    number_states = [2, 3, 4, 5]

aligned_dates_sp = stock_data['Date'][1:].reset_index(drop = True) # aligning length of dates to length of smoothed marginal probabilities
aligned_dates_mp = aligned_dates_sp[1:].reset_index(drop = True) # aligning length of dates to length of model forecasted stock returns

if 'AR' in model_selection: # if AR is in the name of the chosen model
    final_switching_ar = True # allow for switching AR coefficient
else: # otherwise
    final_switching_ar = False # do not allow for switching AR coefficient
    
if 'TREND' in model_selection: # if TREND is in the name of the chosen model
    final_switching_trend = True # allow for switching trend coefficient
else: # otherwise
    final_switching_trend = False # do not allow for switching trend coefficient
    
if 'VARIANCE' in model_selection: # if VARIANCE is in the name of the chosen model
    final_switching_variance = True # allow for switching variance coefficient
else: # otherwise
    final_switching_variance = False # do not allow for switching variance coefficient

for number_state in number_states: # looping over the established number of states

    model = MarkovAutoregression( # creating a Markov switching autoregression model
        endog = stock_data['Stock returns'], # the endogeneous variable is identified with the stock returns
        k_regimes = number_state, # number of regimes
        order = 1, # the order of the autoregressive model is set to be 1, therefore an AR(1) model will be estimated
        switching_ar = final_switching_ar, # taking the parameter from the chosen final model
        switching_trend = final_switching_trend, # taking the parameter from the chosen final model
        switching_variance = final_switching_variance # taking the parameter from the chosen final model
    )

    final_model = model.fit() # fitting the model

# displaying smoothed probabilities of low and high variance regimes for the type of model that we want to choose
fig, axes = plt.subplots(2, figsize = (10,7)) # anticipating 2 subplots with defined figure size
ax = axes[0] # modifying first subplot
ax.plot(aligned_dates_sp, final_model.smoothed_marginal_probabilities[0]) # plotting smoothed marginal probabilities of low-variance regime
ax.grid(False) # removing grid
ax.set(title = 'Smoothed probability of a low-variance regime returns') # title of first subplot
ax = axes[1] # modifying second subplot
ax.plot(aligned_dates_sp, final_model.smoothed_marginal_probabilities[1]) # plotting smoothed marginal probabilities of high-variance regime
ax.set(title = 'Smoothed probability of a high-variance regime returns') # title of second subplot
fig.tight_layout() # tight layout
ax.grid(False) # removing grid
