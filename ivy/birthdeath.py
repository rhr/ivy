"""
Equations from Magallon and Sanderson 2001
"""
from scipy import exp, sqrt, log
from scipy.misc.common import comb

## def alpha(epsilon, r, t):
##     return epsilon*beta(epsilon, r, t)

def Beta(epsilon, r, t):
    exprt = exp(r*t)
    return (exprt - 1)/(exprt - epsilon)

def Alpha(epsilon, r, t):
    return epsilon*Beta(epsilon, r, t)

def prN(i, t, a, r, epsilon):
    """
    i = number of extant species
    t = elapsed time
    a = number of lineages at t=0
    r = net diversification rate (b-d)
    epsilon = relative extinction (d/b)
    """
    beta = Beta(epsilon, r, t)
    alpha = epsilon*beta
    v = sum([ comb(a, j, True)*comb(a+i-j-1, 1, True)*
              pow(alpha, a-j)*pow(beta,i-j)*pow(1-alpha-beta, j)
              for j in range(min(i,a)+1) ])
    return v

def condPrN(i, t, a, r, epsilon):
    """
    conditional probability of i species after time t, given the
    probability of survival
    """
    return prN(i, t, a, r, epsilon)/(1 - pow(Alpha(epsilon, r, t), a))
    
def Nbar(t, a, r, epsilon):
    return (a*exp(r*t)) / (1 - pow(Alpha(epsilon, r, t), a))

def Kendall1948(i, t, r, epsilon):
    """
    prob of i species given single ancestor after time t
    """
    beta = Beta(epsilon, r, t)
    alpha = epsilon*beta
    return (1 - alpha)*(1 - beta)*pow(beta, i-1)

def condKendall1948(i, t, r, epsilon):
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
    log-likelihood of terminal taxa
    t = vector of stem ages
    n = vector of diversities
    r = net diversification
    epsilon = relative extinction
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
