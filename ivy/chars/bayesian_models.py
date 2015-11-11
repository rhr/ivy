import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
from ivy.chars import discrete
import pymc

def create_mk_model(tree, chars, Qtype, pi):
    """
    Create model objects to be passed to pymc.MCMC

    Creates Qparams and likelihood function
    """
    nchar = len(set(chars))

    if Qtype=="ER":
        N = 1
    elif Qtype=="Sym":
        N = int(binom(nchar, 2))
    elif Qtype=="ARD":
        N = int((nchar ** 2 - nchar))
    else:
        ValueError("Qtype must be one of: ER, Sym, ARD")

    # Setting a Dirichlet prior with Jeffrey's hyperprior of 1/2
    print N
    Qparams_init = pymc.Dirichlet(name="probs", theta=([1.0/2.0]*(N+1)))

    # Exponential scaling factor for Qparams
    scaling_factor = pymc.Exponential(name="scaling_factor", beta=1.0)

    @pymc.deterministic(plot=False)
    def Qparams_scaled(q=Qparams_init, s=scaling_factor):
        Qs = np.empty(N)
        for i in range(N):
            Qs[i] = Qparams_init[i]*scaling_factor
        return Qs



    # Qparams_0 = pymc.Exponential(name="Qparams", beta=1.0)
    # Qparams = np.empty(N, dtype=object)
    # Qparams[0] = Qparams_0

    # for i in range(1, N):
    #     Qparams[i] = pymc.Exponential("Qparams_%i" %i, beta=1.0)

    @pymc.potential
    def mklik(q = Qparams_scaled, name="mklik"):
        l = discrete.create_likelihood_function_mk(tree=tree, chars=chars, Qtype=Qtype,
                                      pi="Equal", min=False)
        return l(q)
    return locals()
