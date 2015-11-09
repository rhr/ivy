import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
from ivy.chars import discrete
import pymc3
import pymc


def _create_nodelist(tree, chars):
    """
    Create nodelist. For use in mk function
    """
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nchar = len(set(chars))

    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    nodelist = np.zeros((nnode, nchar+1))
    leafind = [ n.isleaf for n in tree.postiter()]

    for k,ch in enumerate(postChars):
        [ n for i,n in enumerate(nodelist) if leafind[i] ][k][ch] = 1.0
        for i,n in enumerate(nodelist[:-1]):
            n[nchar] = postnodes.index(postnodes[i].parent)

            # Setting initial node likelihoods to one for calculations
    nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

    return nodelist,t

def create_likelihood_function_mk(tree, chars, Qtype, pi="Equal", min=True):
    """
    Create a function that takes values for Q and returns likelihood.

    Specify the Q to be ER, Sym, or ARD

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Qtype (str): What type of Q matrix to use. Either ER (equal rates),
          Sym (symmetric rates), or ARD (All rates different).
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root  node.
    Returns:
        function: Function accepting a list of parameters and returning
          log-likelihood. To be optmimized with scipy.optimize.minimize
    """

    if min:
        nullval = np.inf
    else:
        nullval = -np.inf

    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = range(nchar)

    # Empty Q matrix
    Q = np.zeros([nchar,nchar], dtype=np.double)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    # Empty likelihood array
    nodelist,t = _create_nodelist(tree, chars)
    nodelistOrig = nodelist.copy() # Second copy to refer back to
    # Empty root prior array
    rootpriors = np.empty([nchar], dtype=np.double)

    # Upper bounds
    treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
                   tree.leaves()[0].length])
    upperbound = len(tree.leaves())/treelen

    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used
    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval}

    def likelihood_function(Qparams):
        # Enforcing upper bound on parameters
        if (Qparams > var["upperbound"]) or Qparams <= 0:
            return var["nullval"]

        # Filling Q matrices:
        if Qtype == "ER":
            var["Q"].fill(Qparams)
            var["Q"][np.diag_indices(nchar)] = -Qparams * (nchar-1)
        elif Qtype == "Sym":
            var["Q"].fill(0.0) # Re-filling with zeroes
            xs,ys = np.triu_indices(nchar,k=1)
            var["Q"][xs,ys] = Qparams
            var["Q"][ys,xs] = Qparams
            var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        elif Qtype == "ARD":
            var["Q"].fill(0.0) # Re-filling with zeroes
            s = Qparams.dshape[0]
            print vars(Qparams)
            print vars(Qparams[0])
            print Qparams[1]
            var["Q"][np.triu_indices(nchar, k=1)] = Qparams[:s/2]
            var["Q"][np.tril_indices(nchar, k=-1)] = Qparams[s/2:]
            var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        else:
            raise ValueError, "Qtype must be one of: ER, Sym, ARD"

        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)
        if min:
            x = -1
        else:
            x = 1
        try:
            return x * discrete.mk(tree, chars, var["Q"], p=var["p"], pi = pi, preallocated_arrays=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

# PYMC2

tree = ivy.tree.read("../tests/support/randtree100tipsscale2.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

N=2

Qparams_0 = pymc.Exponential("Qparams", beta=1.0)
Qparams = np.empty(N, dtype=object)
Qparams[0] = Qparams_0

for i in range(1, N):
    Qparams[i] = pymc.Exponential("Qparams_%i" %i, beta=1.0)

@pymc.potential
def lik(q = Qparams):
    l = discrete.create_likelihood_function_mk(tree=tree, chars=chars, Qtype="ARD",
                                  pi="Equal", min=False)
    return l(q)
