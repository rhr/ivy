"""
Equations from Magallon and Sanderson 2001

"""
from __future__ import absolute_import, division, print_function, unicode_literals

from scipy import exp, sqrt, log
from scipy.misc import comb

## def alpha(epsilon, r, t):
##     return epsilon*beta(epsilon, r, t)

def Beta(epsilon, r, t):
    """
    Calculate Beta

    Args:
        epsilon (float): Relative extinction rate(d/b)
        r (float): Net diversification rate (b-d).
        t (float): Elapsed time.
    Returns:
        Float: Beta
    """
    exprt = exp(r*t)
    return (exprt - 1)/(exprt - epsilon)

def Alpha(epsilon, r, t):
    """
    Calculate Alpha

    Args:
        epsilon (float): Relative extinction rate(d/b)
        r (float): Net diversification rate (b-d).
        t (float): Elapsed time
    Returns:
        Float: Alpha
    """
    return epsilon*Beta(epsilon, r, t)

def prN(i, t, a, r, epsilon):
    """
    Probability of observing i species after time t

    Args:
        i (int): Number of extant species
        t (float): Elapsed time
        a (int): Number of lineages at t=0
        r (float): Net diversification rate (b-d)
        epsilon (float):  Relative extinction (d/b)
    """
    beta = Beta(epsilon, r, t)
    alpha = epsilon*beta
    v = sum([ comb(a, j, True)*comb(a+i-j-1, 1, True)*
              pow(alpha, a-j)*pow(beta,i-j)*pow(1-alpha-beta, j)
              for j in range(min(i,a)+1) ])
    return v

def condPrN(i, t, a, r, epsilon):
    """
    Conditional probability of i species after time t, given the
    probability of survival

    Args:
        i (int): Number of extant species
        t (float): Elapsed time
        a (int): Number of lineages at t=0
        r (float): Net diversification rate (b-d)
        epsilon (float):  Relative extinction (d/b)
    """
    return prN(i, t, a, r, epsilon)/(1 - pow(Alpha(epsilon, r, t), a))

def Nbar(t, a, r, epsilon):
    """
    Mean clade size conditional on survival of the clade

    Args:
        t (float): Elapsed time
        a (int): Number of lineages at t=0
        r (float): Net diversification rate (b-d)
        epsilon (float):  Relative extinction (d/b)
    """
    return (a*exp(r*t)) / (1 - pow(Alpha(epsilon, r, t), a))

def Kendall1948(i, t, r, epsilon):
    """
    Probability of observing i species given single ancestor after time t

    Args:
        i (int): Number of extant species
        t (float): Elapsed time
        r (float): Net diversification rate (b-d)
        epsilon (float):  Relative extinction (d/b)
    """
    beta = Beta(epsilon, r, t)
    alpha = epsilon*beta
    return (1 - alpha)*(1 - beta)*pow(beta, i-1)

def condKendall1948(i, t, r, epsilon):
    """
    Probability of observing i species given a single ancestor after time t
    conditional on the clade surviving to time t

    Args:
        i (int): Number of extant species
        t (float): Elapsed time
        r (float): Net diversification rate (b-d)
        epsilon (float):  Relative extinction (d/b)
    """
    beta = Beta(epsilon, r, t)
    return (1 - beta)*pow(beta, i-1)

def r_hat_stem(t, n, epsilon):
    v = (1./t)*log(n*(1-epsilon) + epsilon)
    return v

def r_hat_crown(t, n, epsilon):
    ep2 = epsilon*epsilon
    v = (1./t)*(log(0.5*n*(1-ep2) + 2*epsilon +
                0.5*(1-epsilon)*sqrt(n*(n*ep2 - 8*ep2 + 2*n*epsilon + n)))
                - log(2))
    return v

def logLT(t, n, r, epsilon):
    """
    Log-likelihood of terminal taxa

    Args:
        t: vector of stem ages
        n: vector of diversities
        r (float): net diversification
        epsilon (float): Relative extinction
    """
    v = 0.0
    for ti, ni in zip(t, n):
        rti = r*ti
        exprti = exp(rti)
        bi = (exprti - 1)/(exprti - epsilon)
        A = log(1 - bi)
        B = (ni-1)*log(bi)
        v += A+B
    return v
