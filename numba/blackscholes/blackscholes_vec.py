import numpy as np
import math
import time
import numbapro
from numbapro import vectorize
from numba import autojit, jit
from blackscholes import black_scholes
#import logging; logging.getLogger().setLevel(logging.WARNING)

RISKFREE = 0.02
VOLATILITY = 0.30


@jit('f8(f8)')
def normcdf(d):
    A1 = 0.31938153
    A2 = -0.356563782
    A3 = 1.781477937
    A4 = -1.821255978
    A5 = 1.330274429
    RSQRT2PI = 0.39894228040143267793994605993438
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (RSQRT2PI * math.exp(-0.5 * d * d) *
               (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5))))))
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val

@vectorize(['f8(f8,f8,f8,f8,f8)', 'f4(f4,f4,f4,f4,f4)'])
def black_scholes(S, K, T, R, V):
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1 = normcdf(d1)
    cndd2 = normcdf(d2)
    expRT = math.exp((-1. * R) * T)
    callResult = (S * cndd1 - K * expRT * cndd2)
    putResult = (K * expRT * (1.0 - cndd2) - S * (1.0 - cndd1))
    return callResult

def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


def main(*args):
    OPT_N = 40000    
    callResult = np.zeros(OPT_N)
    putResult = -np.ones(OPT_N)
    stockPrice = randfloat(np.random.random(OPT_N), 5.0, 30.0)
    optionStrike = randfloat(np.random.random(OPT_N), 1.0, 100.0)
    optionYears = randfloat(np.random.random(OPT_N), 0.25, 10.0)

    c = black_scholes(stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY)
    p = None
    return c, p
    
if __name__ == "__main__":
    import sys
    c,p = main(*sys.argv[1:])
