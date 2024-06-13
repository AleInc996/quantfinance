# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 17:00:56 2024

@author: AleInc996
"""

import numpy as np

def options_pricer(S_0, K, T, r, sigma, N, opttype, opttype2, pricingtype):
    
    dt = T / N  # computing time step as the maturity divided by the number of steps
    u = np.exp(sigma * np.sqrt(dt))  # computing up state multiplicator
    d = np.exp(-sigma * np.sqrt(dt))  # computing down state multiplicator
    p_u = (np.exp(r * dt) - d) / (u - d)  # risk neutral probability of up state
    p_d = (u - np.exp(r * dt)) / (u - d) # risk neutral probability of down state
    X = np.zeros([N + 1, N + 1])  # allocating memory for matrix with the evolution of the option price
    S = np.zeros([N + 1, N + 1])  # allocating memory for matrix with the evolution of the underlying price
    delta = np.zeros([N, N])  # allocating memory for matrix with the evolution of delta
    
    for i in range(0, N + 1):
        S[N, i] = S_0 * (u ** (i)) * (d ** (N - i))
        if opttype == "C":
            X[N, i] = max(S[N, i] - K, 0)
        else:
            X[N, i] = max(K - S[N, i], 0)
            
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            X[j, i] = np.exp(-r * dt) * (
                p_u * X[j + 1, i + 1] + (p_d) * X[j + 1, i]
            )  # Computing the European option prices
            S[j, i] = (
                S_0 * (u ** (i)) * (d ** (j - i))
            )  # Underlying evolution for each node
            
            if opttype2 == 'American':
                if opttype == "C":
                    X[j, i] = max(
                        X[j, i], S[j, i] - K
                        )  # Decision between the European option price and the payoff from early-exercise
                else:
                    X[j, i] = max(
                        X[j, i], K - S[j, i]
                        )  # Decision between the European option price and the payoff from early-exercise

            delta[j, i] = (X[j + 1, i + 1] - X[j + 1, i]) / (
                S[j + 1, i + 1] - S[j + 1, i]
            )  # Computing the delta for each node
            
    return X[0, 0], X, S, delta

euro_C, C, S, delta = options_pricer(
    80, 100, 3/12, .05, .2, 1000, "C", "European", "binomial"
)



print("Call option price is {} and his delta at inception is {}".format(round(euro_C,2) ,np.mean(delta)))

euro_P, P, S, Delta = options_pricer(
    100, 100, 3/12, .05, .2, 100, "P", "European", "binomial"
)

print("Put option price is {} and his delta at inception is {}".format(round(euro_P,2) ,np.mean(delta)))





