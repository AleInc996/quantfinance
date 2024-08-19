# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:00:56 2024

@author: AleInc996
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
dt = T/N # computing time step as the maturity divided by the number of steps
opttype = 'Call' # choose between Call and Put
opttype2 = 'European' # choose between European and American
pricingtype = 'Trinomial' # choose between Binomial and Trinomial

def underlying_vector(T, N): # to run only if the idea is to use the trinomial model, the function takes the maturity and number of steps as inputs and computes the evolution of the underlying asset
    
    """
    Defining a function to generate the evolution of the underlying
    asset price in the trinomial case. Creating a function because it is used
    twice in the body of the main function for the trinomial case.
    """
    
    up = np.exp(sigma * np.sqrt(2 * dt)) # computing up multiplicator
    down = np.exp(- sigma * np.sqrt(2 * dt)) # computing down multiplicator

    vector_u = up * np.ones(N) # pre-allocating memory for the upper part of the evolution of the underlying price, filling a vector with up multiplicator
    vector_u = np.cumprod(vector_u)  # computing u, u^2, u^3, ..., u^N

    vector_d = down * np.ones(N) # pre-allocating memory for the down part of the evolution of the underlying price, filling a vector with down multiplicator
    vector_d = np.cumprod(vector_d)  # computing d, d^2, d^3, ..., d^N
    
    total_vector = np.concatenate((vector_d[::-1], [1], vector_u))  # putting together the down part of the evolution, sorting it in ascending order, with 1, that will be multiplied by current underlying price, and the up part of the evolution 
    total_vector = S_0 * total_vector
    return total_vector

def options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype):
    
    if pricingtype == "Binomial":
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
        p_u = ( # defining risk-neutral probability of up movement under trinomial model
            (np.exp(rf * dt / 2) - np.exp(-sigma * np.sqrt(dt / 2)))
            / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        
        p_d = ( # defining risk-neutral probability of down movement under trinomial model
            (-np.exp(rf * dt / 2) + np.exp(sigma * np.sqrt(dt / 2)))
            / (np.exp(sigma * np.sqrt(dt / 2)) - np.exp(-sigma * np.sqrt(dt / 2)))) ** 2
        
        p_m = 1 - p_u - p_d # computing the third risk neutral probability needed in the trinomial case, the middle one between up and down
        
        print(p_u, p_d, p_m) # printing the three risk-neutral probabilities
        
        S = underlying_vector(N, dt) # computing the underlying evolution, change it from calling the function to running it as part of the main function

        if opttype == "Call":
            final_payoff = np.maximum(S - K, 0) # payoff in the case of a call option
        elif opttype == "Put":
            final_payoff = np.maximum(K - S, 0) # payoff in the case of a put option
        else:
            sys.exit("Check that the input for option type is either Call or Put")

        nxt_vec_prices = final_payoff # storing the final payoff vector into a new object
        
        for i in range(1, N + 1):
            stock_vector = underlying_vector(N - i, dt) # creating a new object for editing all observations which require discounting
            expectation = np.zeros(stock_vector.size) # allocating memory for the observations which require discounting
            
            if opttype2 == "European":
                for j in range(expectation.size): # starting from the bottom of each node, multiplying the underlying asset value by the corresponding risk neutral probability
                    tmp = nxt_vec_prices[j] * p_d
                    tmp += nxt_vec_prices[j + 1] * p_m
                    tmp += nxt_vec_prices[j + 2] * p_u

                    expectation[j] = tmp
                nxt_vec_prices = np.exp(-rf * dt) * expectation # the initial final payoff vector is updated with the observations which require discounting actually applying the discount factor
            elif opttype2 == "American":
                for j in range(expectation.size):
                  if opttype == "Call":
                    intrinsic = np.maximum(stock_vector[j] - K, 0) # computing intrinsic value in the case of a call option
                  elif opttype == "Put":
                    intrinsic = np.maximum(K - stock_vector[j], 0) # computing intrinsic value in the case of a put option
                  else:
                    sys.exit("Check that the input for option type is either Call or Put")
                  tmp = nxt_vec_prices[j] * p_d # multipltying underlying asset value by the down risk neutral probability
                  tmp += nxt_vec_prices[j + 1] * p_m # multipltying underlying asset value by the medium risk neutral probability
                  tmp += nxt_vec_prices[j + 2] * p_u # multipltying underlying asset value by the up risk neutral probability

                  expectation[j] = max(intrinsic, tmp * np.exp(-rf * dt)) # the final vector will be the maximum between intrinsic and time value of the option

                nxt_vec_prices = expectation # storing the decisions between intrinsic and time value of the options in the initial variable
                X = nxt_vec_prices # storing nxt_vec_prices in a variable called X, to be consistent with what happens in the Binomial chunk of code

        return X[0], X, S # the expected discounted value of the option at time 0 is the price of the option today, the function will return the price of the option at time 0 (today), the evolution of the option price, the evolution of the underlying asset price and of the delta

    else:
        sys.exit("Check that the input for pricing type is either Binomial or Trinomial")

# to be run for comfort, to have printed outputs
if pricingtype == 'Binomial':
    if opttype == 'Call':
        price_C, C, S, delta = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        print("Call option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_C,2), np.mean(delta)))
    elif opttype == 'Put':
        price_P, P, S, delta = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        print("Put option price, under Binomial model, is {} and his delta at inception is {}".format(round(price_P,2), np.mean(delta)))
elif pricingtype == 'Trinomial':
    if opttype == 'Call':
        price_C, C, S = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        print("Call option price, under Trinomial model, is {}".format(round(price_C,2)))
    elif opttype == 'Put':
        price_P, P, S = options_pricer(S_0, K, T, rf, sigma, N, opttype, opttype2, pricingtype)
        print("Put option price, under Trinomial model, is {}".format(round(price_P,2)))
