# Mk multi regime models
import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
import random

from ivy.chars.mk import *
from ivy.chars.mk_mr import *
from ivy.chars.hrm import *


def mk_multi_regime(tree, chars, Qs, locs, p=None, pi="Fitzjohn", returnPi=False,
                     preallocated_arrays = None):
    """
    Calculate likelhiood of mk model with BAMM-like multiple regimes

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Qs (np.array): Array of instantaneous rate matrices
        locs (np.array): Array of the same length as Qs containing the
          node indices that correspond to each Q matrix
        p (np.array): Optional pre-allocated p matrix
        pi (str or np.array): Option to weight the root node by given values.
           Either a string containing the method or an array
           of weights. Weights should be given in order.

           Accepted methods of weighting root:

           Equal: flat prior
           Equilibrium: Prior equal to stationary distribution
             of Q matrix
           Fitzjohn: Root states weighted by how well they
             explain the data at the tips.
    """
    nchar = Qs[0].shape[0]
    if preallocated_arrays is None:
        # Creating arrays to be used later
        preallocated_arrays = {}
        preallocated_arrays["charlist"] = range(nchar)
        preallocated_arrays["t"] = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)


    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), nchar, nchar], dtype = np.double, order="C")

    inds = [0]*len(preallocated_arrays["t"])

    for l, a in enumerate(locs):
        for n in a:
            inds[n-1] = l

    # Creating probability matrices from Q matrices and branch lengths
    # inds indicates which Q matrix to use for which branch
    cyexpokit.dexpm_treeMulti_preallocated_p(Qs, preallocated_arrays["t"], p, np.array(inds)) # This changes p in place

    if len(preallocated_arrays.keys())==2:
        # Creating more arrays
        nnode = len(tree.descendants())+1
        preallocated_arrays["nodelist"] = np.zeros((nnode, nchar+1))
        leafind = [ n.isleaf for n in tree.postiter()]
        # Reordering character states to be in postorder sequence
        preleaves = [ n for n in tree.preiter() if n.isleaf ]
        postleaves = [n for n in tree.postiter() if n.isleaf ]
        postnodes = list(tree.postiter());prenodes = list(tree.preiter())
        postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
        # Filling in the node list. It contains all of the information needed
        # to calculate the likelihoods at each node
        for k,ch in enumerate(postChars):
            [ n for i,n in enumerate(preallocated_arrays["nodelist"]) if leafind[i] ][k][ch] = 1.0
            for i,n in enumerate(preallocated_arrays["nodelist"][:-1]):
                n[nchar] = postnodes.index(postnodes[i].parent)

        # Setting initial node likelihoods to 1.0 for calculations
        preallocated_arrays["nodelist"][[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

        # Empty array to store root priors
        preallocated_arrays["root_priors"] = np.empty([nchar], dtype=np.double)

    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.cy_mk(preallocated_arrays["nodelist"], p, preallocated_arrays["charlist"])
    # The last row of nodelist contains the likelihood values at the root

    # Applying the correct root prior
    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"

        np.copyto(preallocated_arrays["root_priors"], pi)

        li = sum([ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ])
        logli = math.log(li)

    elif pi == "Equal":
        preallocated_arrays["root_priors"].fill(1.0/nchar)
        li = sum([ float(i)/nchar for i in preallocated_arrays["nodelist"][-1] ])

        logli = math.log(li)

    elif pi == "Fitzjohn":
        np.copyto(preallocated_arrays["root_priors"],
                  [preallocated_arrays["nodelist"][-1,:-1][charstate]/
                   sum(preallocated_arrays["nodelist"][-1,:-1]) for
                   charstate in set(chars) ])

        li = sum([ preallocated_arrays["nodelist"][-1,:-1][charstate] *
                     preallocated_arrays["root_priors"][charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(preallocated_arrays["root_priors"],qsd(Qs[0]))
        li = sum([ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ])
        logli = math.log(li)
    if returnPi:
        return (logli, {k:v for k,v in enumerate(preallocated_arrays["root_priors"])})
    else:
        return logli

def create_likelihood_function_multimk(tree, chars, Qtype, locs, pi="Equal",
                                  min = True):
    if min:
        nullval = np.inf
    else:
        nullval = -np.inf

    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = range(nchar)

    # Empty Q matrix
    Q = np.zeros([len(locs),nchar,nchar], dtype=np.double)
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
           "root_priors":rootpriors, "nullval":nullval, "locs":locs}

    def likelihood_function(Qparams):
        # Enforcing upper bound on parameters

        # TODO: replace with sum of each Q
        if (sum(Qparams)/len(locs) > var["upperbound"]) or any(Qparams <= 0):
            return var["nullval"]

        # Filling Q matrices:
        if Qtype == "ER":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(Qparams[i])
                qmat[np.diag_indices(nchar)] = -Qparams[i] * (nchar-1)
        # elif Qtype == "Sym":
        #     var["Q"].fill(0.0) # Re-filling with zeroes
        #     xs,ys = np.triu_indices(nchar,k=1)
        #     var["Q"][xs,ys] = Qparams
        #     var["Q"][ys,xs] = Qparams
        #     var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        # elif Qtype == "ARD":
        #     var["Q"].fill(0.0) # Re-filling with zeroes
        #     var["Q"][np.triu_indices(nchar, k=1)] = Qparams[:len(Qparams)/2]
        #     var["Q"][np.tril_indices(nchar, k=-1)] = Qparams[len(Qparams)/2:]
        #     var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
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
            return x * mk_multi_regime(tree, chars, var["Q"], var["locs"], p=var["p"], pi = pi, preallocated_arrays=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def create_likelihood_function_multimk_b(tree, chars, Qtype, nregime, pi="Equal",
                                  min = True):
    if min:
        nullval = np.inf
    else:
        nullval = -np.inf

    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = range(nchar)

    # Empty Q matrix
    Q = np.zeros([nregime,nchar,nchar], dtype=np.double)
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

    def likelihood_function(Qparams, locs):
        # Enforcing upper bound on parameters

        # # TODO: replace with sum of each Q
        # if (sum(Qparams)/len(locs) > var["upperbound"]) or any(Qparams <= 0):
        #     return var["nullval"]

        # Filling Q matrices:
        if Qtype == "ER":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(float(Qparams[i]))
                qmat[np.diag_indices(nchar)] = -Qparams[i] * (nchar-1)

        # elif Qtype == "Sym":
        #     var["Q"].fill(0.0) # Re-filling with zeroes
        #     xs,ys = np.triu_indices(nchar,k=1)
        #     var["Q"][xs,ys] = Qparams
        #     var["Q"][ys,xs] = Qparams
        #     var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        elif Qtype == "ARD":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(0.0) # Re-filling with zeroes
                qmat[np.triu_indices(nchar, k=1)] = Qparams[i][:len(Qparams[i])/2]
                qmat[np.tril_indices(nchar, k=-1)] = Qparams[i][len(Qparams[i])/2:]
                qmat[np.diag_indices(nchar)] = 0-np.sum(qmat, 1)
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
            return x * mk_multi_regime(tree, chars, var["Q"], locs, p=var["p"], pi = pi, preallocated_arrays=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def locs_from_switchpoint(tree, switch, locs=None):
    """
    Given a tree and a single node to be the switchpoint, return an
    array of all node indices in one regime vs the other

    Args:
        tree (Node): Root node of tree
        switch (Node): Internal node to be the switchpoint. This node
          is included in its own regime.
    Returns:
        array: Array of all nodes descended from switch, and all other nodes
    """
    # Note: may need to exclude leaves and root
    r1 = [ n.ni for n in switch.preiter()]
    r2 = [ n.ni for n in tree.descendants()if not n.ni in r1 ]

    if locs is None:
        locs = np.empty([2], dtype=object)

    locs[0]=r1
    locs[1]=r2
    return locs
