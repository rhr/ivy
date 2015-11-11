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
    theta = [1.0/2.0]*N
    Qparams_init = pymc.Dirichlet("Qparams_init", theta)
    Qparams_init_full = pymc.CompletedDirichlet("Qparams_init_full", Qparams_init)

    # Exponential scaling factor for Qparams
    scaling_factor = pymc.Exponential(name="scaling_factor", beta=1.0)

    # Scaled Qparams; we would not expect them to necessarily add
    # to 1 as would be the case in a Dirichlet distribution
    @pymc.deterministic(plot=False)
    def Qparams_scaled(q=Qparams_init_full, s=scaling_factor):
        Qs = np.empty(N)
        for i in range(N):
            Qs[i] = q[0][i]*s
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

    
def fit_mk_bayes(tree, chars, Qtype, pi, *kwargs):
    """
    Fit an mk model to a given tree and list of characters. Return
    posterior distributions of Q parameters and MAP estimate of Q matrix

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
        Q: Either a string specifying how to esimate values for Q or a
          numpy array of a pre-specified Q matrix.

          Valid strings for Q:

          "Equal": All rates equal
          "Sym": Forward and reverse rates equal
          "ARD": All rates different
    Keyword Args:
        iters (float): Number of iterations in MCMC. Defaults to 2000
        burn (float): Burnin to discard. Defaults to 200
        thin (float): Thinning parameter. Defaults to 1

    Returns:
        tuple: The pymc MCMC object and the pymc MAP object
    """
    nchar = len(set(chars))
    mod = create_mk_model(tree, chars, Qtype, pi)

    # Arguments for MCMC
    if not kwargs:
        kwargs = {}
    iters = kwargs.pop("iters", 2000)
    burn = kwargs.pop("burn", 200)
    thin = kwargs.pop("thin", 1)

    # MCMC samples
    mc = pymc.MCMC(mod)
    mc.sample(iters, burn, thin)

    # MAP estimation
    mp = pymc.MAP(mod)
    mp.fit()

    return (mc, mp)
