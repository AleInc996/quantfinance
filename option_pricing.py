# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:00:56 2024

@author: AleInc996

Option contracts are derivatives, written on an underlying asset, which give the right (the option) to exercise the contract and 
buy, from the option seller (that is, the option writer) or
sell, to the option buyer (that is, the option holder)
a certain quantity of the underlying asset, at a specified price (strike price) and at a specific point in time (maturity).
The choice of mentioned conditions will depend on the investor's preferences but also on how much the underlying asset is liquid:
the higher the liquidity, the more combinations of strike price and maturity will be provided.
An investor who wants to buy an option will go to the market and select the option that better fits his/her preferences.

Some brief nomenclature follows in order to give context, in case you don't have an options background.
Call options give the right to buy the underlying asset, while put options give the right to sell it. 
An investor who owns a call option believes that the underlying asset price will go up and 
will be profitable if the underlying asset price is higher than strike price + premium (breakeven point),
while an investor who owns a put option believes that the underlying asset price will go down and 
will be profitable if underlying asset price is lower than the strike price.
An investor can be long or short on both call and put options. This means that:
    - if bullish, an investor will go long on a call option or short on a put option
    - if bearish, an investor will go short on a call option or long on a put option

The two most famous types of option are the European one, which can be exercised only at maturity,
and American one, which can be exercised at any time between the beginning of the contract and the maturity date.
Usually, given this, the price of an American option is slightly higher than the one for a European option,
as the long investor needs to pay more for the possibility to have more time to exercise the option.

There are various models built to price options, among which we have the possibility of pricing using binomial or trinomial trees,
where bi- or tri- stands for how many nodes of the trees are generated starting from previous node.
For example, think about time t=0, that is today. From t=0, in a binomial mode two additional nodes are generated for time t=1:
    - up scenario (think of it as if the economy improves from t=0 to t=1)
    - down scenario (think of it as if the economy improves from t=0 to t=1)

In the trinomial model, an additional middle scenario is considered among the up and down ones.

The following script calculates prices, underlying asset price evolution and delta of either a call or put option,
European or American, according to binomial or trinomial model.
Choose the inputs right below and then run the whole script to get results.

"""

import numpy as np
import sys

# inputs for all scenarios of the function options_pricer
S_0 = 90 # underlying asset price at time 0 (today)
K = 100 # strike price
T = 3/12 # maturity
rf = 0.01 # risk-free interest rate
sigma = 0.2 # volatility
N = 1000 # number of steps desired in the construction of the trees
opttype = 'Call' # choose between Call and Put
opttype2 = 'American' # choose between European and American
pricingtype = 'Binomial' # choose between Binomial and Trinomial

def underlying_vector(N, dt): # will be used only if the idea is to use the trinomial model, the function takes the maturity and number of steps as inputs and computes the evolution of the underlying asset
    
    """
    Defining a function to generate the evolution of the underlying
    asset price in the trinomial case. Creating a function because it is used
    twice in the body of the main function for the trinomial case.
    """
    
    u = np.exp(sigma * np.sqrt(2 * dt)) # computing up multiplicator
    d = np.exp(- sigma * np.sqrt(2 * dt)) # computing down multiplicator

    vector_u = u * np.ones(N) # pre-allocating memory for the upper part of the evolution of the underlying price, filling a vector with up multiplicator
    vector_u = np.cumprod(vector_u)  # computing u, u^2, u^3, ..., u^N

    vector_d = d * np.ones(N) # pre-allocating memory for the down part of the evolution of the underlying price, filling a vector with down multiplicator
    vector_d = np.cumprod(vector_d)  # computing d, d^2, d^3, ..., d^N
    
    total_vector = np.concatenate((vector_d[::-1], [1], vector_u))  # putting together the down part of the evolution, sorting it in ascending order, with 1, that will be multiplied by current underlying price, and the up part of the evolution 
    total_vector = S_0 * total_vector
    return total_vector

def options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype):
    
    if pricingtype == "Binomial":
        dt = T / N # computing time step as the maturity divided by the number of steps
        u = np.exp(sigma * np.sqrt(dt))  # computing up state multiplicator (multiplicator for an up movement)
        d = np.exp(-sigma * np.sqrt(dt))  # computing down state multiplicator (multiplicator for a down movement))
        p_u = (np.exp(rf * dt) - d) / (u - d)  # risk neutral probability of up state
        p_d = (u - np.exp(rf * dt)) / (u - d) # risk neutral probability of down state
        X = np.zeros([N + 1, N + 1])  # allocating memory for matrix with the evolution of the option price
        S = np.zeros([N + 1, N + 1])  # allocating memory for matrix with the evolution of the underlying price
        delta = np.zeros([N, N])  # allocating memory for matrix with the evolution of delta, one dimension less than price and underlying because delta at step 0 refers to price and underlying at step 1 and so on
        
        for i in range(0, N + 1): # this for loop block will create the last steps of the binomial tree, while the next for block will fill the tree backwards
        
            S[N, i] = S_0 * (u ** (i)) * (d ** (N - i)) # filling the last row of the matrix showing the evolution of the underlying price
            if opttype == "Call":
                X[N, i] = max(S[N, i] - K, 0) # if call option, filling the last row of the matrix showing the evolution of the option price
            elif opttype == "Put":
                X[N, i] = max(K - S[N, i], 0) # if put option, filling the last row of the matrix showing the evolution of the option price
            else:
                print('Please check that the type of option selected is in line with available choices')
                break
        
        for j in range(N - 1, -1, -1):
            for i in range(0, j + 1): # the evolutions of both option price and underlying price will be filled starting from left bottom part of the matrices
                
                X[j, i] = np.exp(-rf * dt) * (p_u * X[j + 1, i + 1] + (p_d) * X[j + 1, i]) # computing the European option prices
                S[j, i] = (S_0 * (u ** (i)) * (d ** (j - i))) # computing the evolution of the underlying for each node
                
                if opttype2 == 'American':
                    if opttype == "Call": # American call case
                        X[j, i] = max(X[j, i], S[j, i] - K)  # for an American option, the decision is between the European option price and the payoff obtained if we exercise the option early
                    elif opttype == "Put": # American put case
                        X[j, i] = max(X[j, i], K - S[j, i])  # for an American option, the decision is between the European option price and the payoff obtained if we exercise the option early
                    else:
                        print('Please check that the type of option selected is in line with available choices')
                        break
                
                delta[j, i] = (X[j + 1, i + 1] - X[j + 1, i]) / (S[j + 1, i + 1] - S[j + 1, i]) # computing the delta for each node
                
        return X[0, 0], X, S, delta # the function will return the price of the option at time 0 (today), the evolution of the option price, the evolution of the underlying asset price and of the delta

    elif pricingtype == "Trinomial":
        dt = T / N # computing time step as the maturity divided by the number of steps
        
        p_u = ((np.exp(rf * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2)))
            / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2 # defining risk-neutral probability of up movement under trinomial model
        
        p_d = ((-np.exp(rf * dt / 2) + np.exp(sigma * np.sqrt(dt / 2)))
            / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2 # defining risk-neutral probability of down movement under trinomial model
        
        p_m = 1 - p_u - p_d # computing the third risk neutral probability needed in the trinomial case, the middle one between up and down
        
        print(p_u, p_d, p_m) # printing the three risk-neutral probabilities
        
        S = underlying_vector(N, dt) # computing the underlying evolution, calling the function created above
        
        # the following payoffs are considered if European is the choice for opttype2
        if opttype == "Call":
            final_payoff = np.maximum(S - K, 0) # payoff in the case of a call option
        elif opttype == "Put":
            final_payoff = np.maximum(K - S, 0) # payoff in the case of a put option
        else:
            sys.exit("Check that the input for option type is either Call or Put")

        prices_vector = final_payoff # storing the final payoff vector into a new object
        
        for i in range(1, N + 1):
            stock_vector = underlying_vector(N - i, dt) # creating a new object for editing all observations which require discounting
            expectation_vector = np.zeros(stock_vector.size) # allocating memory for the observations which require discounting
            
            if opttype2 == "European":
                for j in range(expectation_vector.size): # starting from the bottom of each node, multiplying the underlying asset value by the corresponding risk neutral probability
                    temp = prices_vector[j] * p_d
                    temp += prices_vector[j + 1] * p_m
                    temp += prices_vector[j + 2] * p_u

                    expectation_vector[j] = temp
                prices_vector = np.exp(-rf * dt) * expectation_vector # the initial final payoff vector is updated with the observations which require discounting actually applying the discount factor
            elif opttype2 == "American": # when it comes to American options, we need to distinguish between intrinsic and time values of the option
                for j in range(expectation_vector.size):
                  if opttype == "Call":
                    intrinsic_value = np.maximum(stock_vector[j] - K, 0) # computing intrinsic value in the case of a call option
                  elif opttype == "Put":
                    intrinsic_value = np.maximum(K - stock_vector[j], 0) # computing intrinsic value in the case of a put option
                  else:
                    sys.exit("Check that the input for option type is either Call or Put")
                  temp = prices_vector[j] * p_d # multipltying underlying asset value by the down risk neutral probability
                  temp += prices_vector[j + 1] * p_m # multipltying underlying asset value by the medium risk neutral probability
                  temp += prices_vector[j + 2] * p_u # multipltying underlying asset value by the up risk neutral probability

                  expectation_vector[j] = max(intrinsic_value, np.exp(-rf * dt) * temp) # the final vector will be the maximum between intrinsic and time value of the option

                prices_vector = expectation_vector # storing the decisions between intrinsic and time value of the options in the initial variable
        X = prices_vector # storing nxt_vec_prices in a variable called X, to be consistent with what happens in the Binomial chunk of code

        return X[0], X, S # the expected discounted value of the option at time 0 is the price of the option today, the function will return the price of the option at time 0 (today), the evolution of the option price, the evolution of the underlying asset price and of the delta

    else:
        sys.exit("Check that the input for pricing type is either Binomial or Trinomial")

# the following chunk of code is to be run for comfort, to have printed outputs
if pricingtype == 'Binomial':
    if opttype == 'Call':
        price_C, C, S, delta = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        if opttype2 == "European":
            print("European Call option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_C,2), np.mean(delta)))
        elif opttype2 == "American":
            print("American Call option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_C,2), np.mean(delta)))
    elif opttype == 'Put':
        price_P, P, S, delta = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        if opttype2 == "European":
            print("European Put option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_P,2), np.mean(delta)))
        elif opttype2 == "American":
            print("American Put option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_P,2), np.mean(delta)))
elif pricingtype == 'Trinomial':
    if opttype == 'Call':
        price_C, C, S = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        if opttype2 == "European":
            print("European Call option price, under Trinomial model, is {}".format(round(price_C,2)))
        elif opttype2 == "American":
            print("American Call option price, under Trinomial model, is {}".format(round(price_C,2)))
    elif opttype == 'Put':
        price_P, P, S = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        if opttype2 == "European":
            print("European Put option price, under Trinomial model, is {}".format(round(price_P,2)))
        elif opttype2 == "American":
            print("American Put option price, under Trinomial model, is {}".format(round(price_P,2)))
