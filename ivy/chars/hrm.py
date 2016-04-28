#!/usr/bin/env python

import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
import itertools
import random
import multiprocessing as mp
from functools import partial
import pickle

from ivy.chars.mk import *
from ivy.chars.mk_mr import *
from ivy.chars.hrm import *
from scipy import cluster
import nlopt
np.seterr(invalid="warn")

"""
Functions for fitting an HRM model
"""


def hrm_mk(tree, chars, Q, nregime, pi="Equal",returnPi=False,
          ar=None):
    """
    Return log-likelihood of hidden-rates-markov mk model as described in
    Beaulieu et al. 2013

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Q (np.array): Instantaneous rate matrix
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
        returnPi (bool): Whether or not to return the final values of root
          node weighting
        ar (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays
    """
    nchar = Q.shape[0]
    nobschar = nchar/nregime
    if ar is None:
        # Creating arrays to be used later
        ar = create_hrm_ar(tree, chars, nregime)
    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.dexpm_tree_preallocated_p_log(Q, ar["t"], ar["p"])
    cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"])
    # The last row of nodelist contains the likelihood values at the root

    # Applying the correct root prior
    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"
        np.copyto(ar["root_priors"], pi)
        rootliks = [ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ]
    elif pi == "Equal":
        ar["root_priors"].fill(1.0/nchar)
        rootliks = [ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1])]
    elif pi == "Fitzjohn":
        np.copyto(ar["root_priors"],
                  [ar["nodelist"][-1,:-1][charstate]-
                   scipy.misc.logsumexp(ar["nodelist"][-1,:-1]) for
                   charstate in range(nchar) ])
        rootliks = [ ar["nodelist"][-1,:-1][charstate] +
                     ar["root_priors"][charstate] for charstate in range(nchar) ]
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(ar["root_priors"],qsd(Q))
        rootliks = [ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ]
    logli = scipy.misc.logsumexp(rootliks)

    if returnPi:
        return (logli, {k:v for k,v in enumerate(ar["root_priors"])})
    else:
        return logli


def create_likelihood_function_hrm_mk(tree, chars, nregime, Qtype, pi="Fitzjohn",
                                  findmin = True):
    """
    Create a function that takes values for Q and returns likelihood.

    Specify the Q to be ER, Sym, or ARD

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Qtype (str): ARD only
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root  node.
        min (bool): Whether the function is to be minimized (False means
          it will be maximized)
    Returns:
        function: Function accepting a list of parameters and returning
          log-likelihood. To be optmimized with scipy.optimize.minimize
    """
    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))
    var = create_hrm_ar(tree, chars, nregime, findmin)
    def likelihood_function(Qparams):
        # Enforcing upper bound on parameters
        if (sum(Qparams) > (var["upperbound"]*2)) or any(Qparams <= 0):
            return var["nullval"]

        # Filling Q matrices:
        i = 0
        if Qtype == "ARD":
            var["Q"].fill(0.0) # Re-filling with zeroes
            for rC in range(nregime):
                for charC in range(nobschar):
                  for rR in range(nregime):
                        for charR in range(nobschar):
                            if not ((rR == rC) and (charR == charC)):
                                if ((rR == rC) or ((charR == charC)) and (rR+1 == rC or rR-1 == rC)):
                                    var["Q"][charR+rR*nobschar, charC+rC*nobschar] = Qparams[i]
                                    i += 1
            var["Q"][np.diag_indices(nchar)] = np.sum(var["Q"], axis=1)*-1
        elif Qtype == "Simple":
            var["Q"].fill(0.0)
            fill_Q_matrix(nobschar, nregime, Qparams[:nregime], Qparams[nregime:], Qtype="Simple", out = var["Q"])
        else:
            raise ValueError, "Qtype must be ARD or Simple"
        for char in range(nobschar):
            hiddenchar =  [y + char for y in [x * nobschar for x in range(nregime) ]]
            for char2 in [ ch for ch in range(nobschar) if not ch == char ]:
                hiddenchar2 =  [y + char2 for y in [x * nobschar for x in range(nregime) ]]
                rs = [var["Q"][ch1, ch2] for ch1, ch2 in zip(hiddenchar, hiddenchar2)]
                if not(rs == sorted(rs)):
                    return var["nullval"]
        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)
        if findmin:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_mk(tree, chars, var["Q"],nregime, pi = pi, ar=var)
            if not np.isnan(logli):
                return x * logli# Minimizing negative log-likelihood
            else:
                return var["nullval"]
        except ValueError: # If likelihood returned is 0
            return var["nullval"]
    return likelihood_function


def _random_Q_matrix(nobschar, nregime):
    """
    Generate a random Q matrix with nchar*nregime rows and cols
    """
    split = 2.0/nregime
    bins = [ split*r for r in range(nregime) ]
    Q = np.zeros([nobschar*nregime, nobschar*nregime])
    for rC in range(nregime):
        for charC in range(nobschar):
          for rR in range(nregime):
                for charR in range(nobschar):
                    if not ((rR == rC) and (charR == charC)):
                        if ((rR == rC) or ((charR == charC)) and (rR+1 == rC or rR-1 == rC)):
                            Q[charR+rR*nobschar, charC+rC*nobschar] = random.uniform(bins[rR], bins[rR]+split)
    Q[np.diag_indices(nobschar*nregime)] = np.sum(Q, axis=1)*-1
    return Q


def fill_Q_matrix(nobschar, nregime, wrparams, brparams, Qtype="ARD", out=None, orderedRegimes = True):
    """
    Fill a Q matrix with nchar*nregime rows and cols with values from Qparams

    Args:
        nchar (int): number of observed characters
        nregime (int): number of hidden rates per character
        wrparams (list): List of unique Q values for within-regime transitions,
          in order as they appear in columnwise iteration
        brparams (list): list of unique Q values for between-regime transition,
          in order as they appear in columnwise iteration
        Qtype (str): Either "ARD" or "Simple". What type of Q matrix to fill.
        out (np.array): Optional numpy array to put output into
        orderedRegimes (bool): Whether or not regimes are ordered such that
          a transition from 1 -> 3 is not possible (a branch in state 1 must
          pass through 2 to get to 3)
    Returns:
        array: Q-matrix with values filled in. Check to make sure values
          have been filled in properly
    """
    wrparams = list(wrparams)
    brparams = list(brparams)
    if out is None:
        Q = np.zeros([nobschar*nregime, nobschar*nregime])
    else:
        Q = out
    assert Qtype in ["ARD", "Simple"]
    if orderedRegimes:
        grid = np.zeros([(nobschar*nregime)**2, 4], dtype=int)
        grid[:,0] = np.tile(np.repeat(range(nregime), nobschar), nobschar*nregime)
        grid[:,1] = np.repeat(range(nregime), nregime*nobschar**2)
        grid[:,2] = np.tile(range(nobschar), nregime**2*nobschar)
        grid[:,3] = np.tile(np.repeat(range(nobschar), nregime*nobschar), nregime)
        if Qtype == "ARD":
            wrcount = 0
            brcount = 0
            for i, qcell in enumerate(np.nditer(Q, order="F", op_flags=["readwrite"])):
                cell = grid[i]
                if (cell[0] == cell[1]) and cell[2] != cell[3]:
                    qcell[...] = wrparams[wrcount]
                    wrcount += 1
                elif(cell[0] in [cell[1]+1, cell[1]-1] and cell[2] == cell[3] ):
                    qcell[...] = brparams[brcount]
                    brcount += 1
        elif Qtype == "Simple":
            for i,qcell in enumerate(np.nditer(Q, order="F", op_flags=["readwrite"])):
                cell = grid[i]
                if (cell[0] == cell[1]) and cell[2] != cell[3]:
                    qcell[...] = wrparams[cell[0]]
                elif(cell[0] in [cell[1]+1, cell[1]-1] and cell[2] == cell[3] ):
                    qcell[...] = brparams[0]
    else:
        n_wr = nobschar**2-nobschar
        for i,wr in enumerate([wrparams[i:i+n_wr] for i in xrange(0, len(wrparams), n_wr)]):
            subQ = slice(i*nobschar,(i+1)*nobschar)
            wrVals = [x for s in [[0]+wr[k:k+nobschar] for k in range(0, len(wr)+1, nobschar)] for x in s]
            np.copyto(Q[subQ,subQ], [wrVals[x:x+nobschar] for x in xrange(0, len(wrVals), nobschar)])
        combs = list(itertools.combinations(range(nregime),2))
        revcombs = [tuple(reversed(i)) for i in combs]
        submatrix_indices = [x for s in [[combs[i]] + [revcombs[i]] for i in range(len(combs))] for x in s]
        for i,submatrix_index in enumerate(submatrix_indices):
            my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
            my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
            nregimeswitch = (nregime**2 - nregime)*nobschar
            np.fill_diagonal(Q[my_slice0,my_slice1],[p for p in brparams[i*nobschar:i*nobschar+nobschar]])
    Q[np.diag_indices(nobschar*nregime)] = (np.sum(Q, axis=1) * -1)
    if out is None:
        return Q


def n_Qparams(nchar, nregime):
    """
    Number of free Q params for a matrix with nchar and nregimes
    """
    Cs = [ (i,j) for i in range(nregime) for j in range(nchar)]
    n = [(i,j) for i in Cs for j in Cs if (i!=j and (i[0] + 1 == j[0] or i[0] - 1 == j[0] or i[0]==j[0]) and (i[0]==j[0] or i[1] == j[1])) ]
    return(len(n))


def create_hrm_ar(tree, chars, nregime, findmin=True):
    """
    Create arrays to be used in hrm likelihood function
    """
    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nchar = len(set(chars)) * nregime
    nobschar = len(set(chars))
    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    nodelist = np.zeros((nnode, nchar+1))
    nodelist.fill(-np.inf) # Fill initial likelihoods with log(0) (-inf)
    childlist = np.zeros(nnode, dtype=object)
    leafind = [ n.isleaf for n in tree.postiter()]
    tmp_ar = np.zeros([nchar]) # For storing calculations in cython code

    for k,ch in enumerate(postChars):
        hiddenChs = [y + ch for y in [x * nobschar for x in range(nregime) ]]
        [ n for i,n in enumerate(nodelist) if leafind[i] ][k][hiddenChs] = np.log(1.0)
    for i,n in enumerate(nodelist[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)
        childlist[i] = [ nod.pi for nod in postnodes[i].children ]
    childlist[i+1] = [ nod.pi for nod in postnodes[i+1].children ] # Add the root to the childlist array

    # Setting initial node likelihoods to log one for calculations
    nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = np.log(1.0)
    # Empty Q matrix
    Q = np.zeros([nchar,nchar], dtype=np.double)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    nodelistOrig = nodelist.copy() # Second copy to refer back to
    # Empty root prior array
    rootpriors = np.empty([nchar], dtype=np.double)
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf
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
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar}
    return var


def hrm_back_mk(tree, chars, Q, nregime, p=None, pi="Fitzjohn",returnPi=False,
                ar=None, tip_states=None, returnnodes=False):
    """
    Calculate probability vector at root given tree, characters, and Q matrix,
    then reconstruct probability vectors for tips and use those in another
    up-pass to calculate probability vector at root.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Q (np.array): Instantaneous rate matrix
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
        returnPi (bool): Whether or not to return the final values of root
          node weighting
        ar (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays
    """
    nchar = Q.shape[0]
    nobschar = nchar/nregime
    if ar is None:
        # Creating arrays to be used later
        ar = {}
        t = [node.length for node in tree.postiter() if not node.isroot]
        t = np.array(t, dtype=np.double)
        ar["charlist"] = range(Q.shape[0])
        ar["t"] = t

    if p is None: # Instantiating empty array
        p = np.empty([len(ar["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, ar["t"], p) # This changes p in place

    if len(ar.keys())==2:
        # Creating more arrays
        nnode = len(tree.descendants())+1
        ar["nodelist"] = np.zeros((nnode, nchar+1))
        ar["childlist"] = np.zeros(nnode, dtype=object)
        leafind = [ n.isleaf for n in tree.postiter()]
        # Reordering character states to be in postorder sequence
        preleaves = [ n for n in tree.preiter() if n.isleaf ]
        postleaves = [n for n in tree.postiter() if n.isleaf ]
        postnodes = list(tree.postiter());prenodes = list(tree.preiter())
        postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
        # Filling in the node list. It contains all of the information needed
        # to calculate the likelihoods at each node

        # Q matrix is in the form of "0S, 1S, 0F, 1F" etc. Probabilities
        # set to 1 for all hidden states of the observed state.
        for k,ch in enumerate(postChars):
            # Indices of hidden rates of observed state. These will all be set to 1
            hiddenChs = [y + ch for y in [x * nobschar for x in range(nregime) ]]
            [ n for i,n in enumerate(ar["nodelist"]) if leafind[i] ][k][hiddenChs] = 1.0/nregime
        for i,n in enumerate(ar["nodelist"][:-1]):
            n[nchar] = postnodes.index(postnodes[i].parent)
            ar["childlist"][i] = [ nod.pi for nod in postnodes[i].children ]
        ar["childlist"][i+1] = [ nod.pi for nod in postnodes[i+1].children ]

        # Setting initial node likelihoods to 1.0 for calculations
        ar["nodelist"][[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

        # Empty array to store root priors
        ar["root_priors"] = np.empty([nchar], dtype=np.double)
        ar["nodelist-up"] = ar["nodelist"].copy()
        ar["t_Q"] = Q
        ar["p_up"] = p.copy()
        ar["v"] = np.zeros([nchar])
        ar["tmp"] = np.zeros([nchar+1])
        ar["motherRow"] = np.zeros([nchar+1])

    leafind = [ n.isleaf for n in tree.postiter()]
    if tip_states is not None:
        leaf_rownums = [i for i,n in enumerate(leafind) if n]
        tip_states = ar["nodelist"][leaf_rownums][:,:-1] * tip_states[:,:-1]
        tip_states = tip_states/np.sum(tip_states,1)[:,None]

        ar["nodelist"][leaf_rownums,:-1] = tip_states

    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.cy_mk(ar["nodelist"], p, np.array(ar["charlist"]))
    # The last row of nodelist contains the likelihood values at the root
    # Applying the correct root prior
    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"

        np.copyto(ar["root_priors"], pi)

        li = sum([ i*ar["root_priors"][n] for n,i in enumerate(ar["nodelist"][-1,:-1]) ])
        logli = math.log(li)

    elif pi == "Equal":
        ar["root_priors"].fill(1.0/nchar)
        li = sum([ float(i)/nchar for i in ar["nodelist"][-1] ])

        logli = math.log(li)

    elif pi == "Fitzjohn":
        np.copyto(ar["root_priors"],
                  [ar["nodelist"][-1,:-1][charstate]/
                   sum(ar["nodelist"][-1,:-1]) for
                   charstate in range(nchar) ])

        li = sum([ ar["nodelist"][-1,:-1][charstate] *
                     ar["root_priors"][charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(ar["root_priors"],qsd(Q))
        li = sum([ i*ar["root_priors"][n] for n,i in enumerate(ar["nodelist"][-1,:-1]) ])
        logli = math.log(li)

    # Transposal of Q for up-pass now that down-pass is completed
    np.copyto(ar["t_Q"], Q)
    ar["t_Q"] = np.transpose(ar["t_Q"])
    ar["t_Q"][np.diag_indices(nchar)] = 0
    ar["t_Q"][np.diag_indices(nchar)] = -np.sum(ar["t_Q"], 1)
    ar["t_Q"] = np.ascontiguousarray(ar["t_Q"])
    cyexpokit.dexpm_tree_preallocated_p(ar["t_Q"], ar["t"], ar["p_up"])
    ar["nodelist-up"][:,:-1] = 1.0
    ar["nodelist-up"][-1] = ar["nodelist"][-1]

    ni = len(ar["nodelist-up"]) - 2

    root_marginal =  ivy.chars.mk.qsd(Q) # Change to Fitzjohn Q?

    for n in ar["nodelist-up"][::-1][1:]:
        curRow = n[:-1]
        motherRowNum = int(n[nchar])
        np.copyto(ar["motherRow"], ar["nodelist-up"][int(motherRowNum)])
        sisterRows = [ (ar["nodelist-up"][i],i) for i in ar["childlist"][motherRowNum] if not i==ni]

        # If the mother is the root...
        if ar["motherRow"][nchar] == 0.0:
            # The marginal of the root
            np.copyto(ar["v"],root_marginal) # Only need to calculate once
        else:
            # If the mother is not the root, calculate prob. of being in any state
            # Use transposed matrix
            np.dot(ar["p_up"][motherRowNum], ar["nodelist-up"][motherRowNum][:nchar], out=ar["v"])
        for s in sisterRows:
            # Use non-transposed matrix
            np.copyto(ar["tmp"], ar["nodelist"][s[1]])
            ar["tmp"][:nchar] = ar["tmp"][:-1]/sum(ar["tmp"][:nchar])
            ar["v"] *= np.dot(p[s[1]], ar["tmp"][:nchar])
        ar["nodelist-up"][ni][:nchar] = ar["v"]
        ni -= 1
    out = [ar["nodelist-up"][[ t.pi for t in tree.leaves() ]], logli]
    if returnnodes:
        out.append(ar["nodelist-up"])
    return out


def create_likelihood_function_hrm_mk_MLE(tree, chars, nregime, Qtype, pi="Fitzjohn",
                                  findmin = True, constraints = "Rate",
                                  orderedRegimes = True):
    """
    Create a function that takes values for Q and returns likelihood.

    Specify the Q to be ER, Sym, or ARD

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Qtype (str): ARD only
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root  node.
        min (bool): Whether the function is to be minimized (False means
          it will be maximized)
        orderedRegimes (bool): Whether or not regimes are ordered such that
          a transition from 1 -> 3 is not possible (a branch in state 1 must
          pass through 2 to get to 3)
    Notes:
        Constraints:
            The maximum rate within each rate category must be ordered (
            fastest rate in slower regime must be slower than fastest rate
            in fastest regime)
    Returns:
        function: Function accepting a list of parameters and returning
          log-likelihood. To be optmimized with scipy.optimize.minimize
    """
    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))
    var = create_hrm_ar(tree, chars, nregime,findmin)
    def likelihood_function(Qparams, grad=None):
        """
        NLOPT inputs the parameter array as well as a gradient object.
        The gradient object is ignored for this optimizer (LN_SBPLX)
        """
        if any(np.isnan(Qparams)):
            return var["nullval"]
        # Enforcing upper bound on parameters
        if (sum(Qparams) > (var["upperbound"]*2)) or any(Qparams <= 0):
            return var["nullval"]
        if Qtype == "ARD":
            if constraints == "Rate":
                qmax = [max(Qparams[i:i+nobschar]) for i in xrange(0, len(Qparams)/2, nobschar)]
                if sorted(qmax) != qmax:
                    return var["nullval"]
            elif constraints == "Symmetry":
                # TODO: generalize this
                assert len(Qparams) == 8
                if not ((Qparams[0]/Qparams[1] <= 1) and (Qparams[2]/Qparams[3] >= 1)):
                    return var["nullval"]
            elif constraints == "corHMM":
                assert len(Qparams) == 8
                if not Qparams[1] < Qparams[3]:
                    return var["nullval"]
        elif Qtype == "Simple":
            if any(sorted(Qparams[:-1]) != Qparams[:-1]):
                return var["nullval"]
            if Qparams[-2] < Qparams[-1]:
                return var["nullval"]
        # Filling Q matrices:
        i = 0
        if Qtype == "ARD":
            var["Q"].fill(0.0) # Re-filling with zeroes
            wr = ((nobschar**2-nobschar)*nregime)
            fill_Q_matrix(nobschar, nregime, Qparams[:wr], Qparams[wr:],
                          Qtype="ARD", out=var["Q"], orderedRegimes=orderedRegimes)
        # TODO: orderedRegimes for simple
        elif Qtype == "Simple":
            var["Q"].fill(0.0)
            fill_Q_matrix(nobschar, nregime, Qparams[:nregime], Qparams[nregime:], Qtype="Simple", out = var["Q"],
                          orderedRegimes=False)
        else:
            raise ValueError, "Qtype must be ARD or Simple"
        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if findmin:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_mk(tree, chars, var["Q"],nregime, pi = pi, ar=var)
            if not np.isnan(logli):
                return x * logli# Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]
    return likelihood_function


def fit_hrm(tree, chars, nregime, pi="Equal", constraints="Rate", Qtype="Simple"):
    """
    Fit a hidden-rates mk model to a given tree and list of characters, and
    number of regumes. Return fitted ARD Q matrix and calculated likelihood.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        nregime (int): Number of hidden rates per character
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
    Returns:
        tuple: Tuple of fitted Q matrix (a np array) and log-likelihood value
    """
    if Qtype == "Simple":
        return fit_hrm_mkSimple(tree, chars, nregime, pi)
    elif Qtype == "ARD":
        return fit_hrm_mkARD(tree, chars, nregime, pi, constraints)
    else:
        raise TypeError, "Qtype must be Simple or ARD"


def fit_hrm_mkARD(tree, chars, nregime, pi="Fitzjohn", constraints="Rate",
                  orderedRegimes = True):
    """
    Fit a hidden-rates mk model to a given tree and list of characters, and
    number of regumes. Return fitted ARD Q matrix and calculated likelihood.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        nregime (int): Number of hidden rates per character
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
        orderedRegimes (bool): Whether or not to constrain regime transitions
          to only occur between adjacent regimes
    Returns:
        tuple: Tuple of fitted Q matrix (a np array) and log-likelihood value
    """
    nchar = len(set(chars))*nregime
    nobschar = len(set(chars))
    mk_func = create_likelihood_function_hrm_mk_MLE(tree, chars, nregime=nregime,
                                                 Qtype="ARD", pi=pi, constraints=constraints,
                                                 orderedRegimes=orderedRegimes)
    ncell = ((nobschar**2-nobschar)*nregime + (nregime**2-nregime)*nobschar)
    x0 = [0.1]*ncell
    opt = nlopt.opt(nlopt.LN_SBPLX, ncell)
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)
    optim = opt.optimize(x0)


    wr = (nobschar**2-nobschar)*nregime
    q = fill_Q_matrix(nobschar, nregime, optim[:wr], optim[wr:],"ARD", orderedRegimes=orderedRegimes)
    piRates = hrm_mk(tree, chars, q, nregime, pi=pi, returnPi=True)[1]
    return (q, -1*float(mk_func(optim, None)), piRates)

def fit_hrm_mkSimple(tree, chars, nregime, pi="Fitzjohn"):
    """
    Fit a hidden-rates mk model to a given tree and list of characters, and
    number of regumes. Return fitted ARD Q matrix and calculated likelihood.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        nregime (int): Number of hidden rates per character
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
    Returns:
        tuple: Tuple of fitted Q matrix (a np array) and log-likelihood value
    """
    nchar = len(set(chars))*nregime
    nobschar = len(set(chars))
    mk_func = create_likelihood_function_hrm_mk_MLE(tree, chars, nregime=nregime,
                                                 Qtype="Simple", pi=pi)
    x0 = [0.1] * (nregime+1)
    opt = nlopt.opt(nlopt.LN_SBPLX, nregime+1)
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)
    optim = opt.optimize(x0)

    q = fill_Q_matrix(nobschar, nregime, optim[:-1], [optim[-1]],"Simple")
    piRates = hrm_mk(tree, chars, q, nregime, pi=pi, returnPi=True)[1]
    return (q, -1*float(mk_func(optim, None)), piRates)


def make_regime_type_combos(nregime, nparams):
    """
    Create regime combinations for a binary character
    """
    paramRanks = range(nparams+1)
    paramPairs = list(itertools.permutations(paramRanks,2))
    for p in paramRanks:
        paramPairs.append((p,p))

    regimePairs = list(itertools.combinations(paramPairs,nregime))
    return regimePairs


def remove_redundant_regimes(regimePairs):
    """
    Remove redundant regime pairs (eg (1,1),(2,2) and (2,2),(3,3) are
    redundant and only one should be kept)
    """
    regime_identities = [make_identity_regime(r) for r in regimePairs]
    return [regimePairs[regimePairs.index(i)] for i in set(regime_identities)]


def make_identity_regime(regimePair):
    """
    Given one regime pair, reduce it to its identity (eg (2,2),(3,3) becomes
    (1,1),(2,2)
    """
    params_present = list(set([i for s in regimePair for i in s]))
    if 0 in params_present:
        identity_params = range(len(params_present))
    else:
        identity_params = [i+1 for i in range(len(params_present))]
    identity_regime =  tuple([tuple([identity_params[params_present.index(i)]
                                     for i in r]) for r in regimePair])
    return identity_regime


def optimize_regime(comb, tree=None, chars=None, nregime=None, nparams=None,
                    pi=None, br_variable=None, out_file=None, ar=None):
    """
    Top-level function for optimizing hrm distinct-regime model in parallel.
    """
    mk_func = hrm_disctinct_regimes_likelihoodfunc(tree, chars, comb, pi=pi,
            findmin = True, br_variable=br_variable, ar=ar)
    if not br_variable:
        nfreeparams = nparams+1
    else:
        nfreeparams = nparams + (nregime**2 - nregime)*2
    x0 = [0.1] * nfreeparams
    opt = nlopt.opt(nlopt.LN_SBPLX, nfreeparams)
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)
    pars = opt.optimize(x0)

    lik = mk_func(pars)
    Q = fill_distinct_regime_Q(comb, np.insert(pars, 0, 1e-15),nregime,2,br_variable=br_variable)
    if out_file is not None:
        with open(out_file+str(comb)+".p", "wb") as f:
            pickle.dump((comb, lik, Q), f)
    return comb,lik,Q


def fit_hrm_distinct_regimes(tree, chars, nregime, nparams, pi="Equal", br_variable=False,
                             parallel=False, ncores=2, out_file=None):
    """
    Fit hrm with distinct regime types given number of regimes and
    number of parameters

    BINARY CHARACTERS ONLY

    Args:
        tree (Node): Root node of tree
        chars (list): Character states in preorder sequence
        nregime (int): Number of regimes to test in model
        nparams (int): Number of unique parameters available for regimes
        pi (str): Root prior
        br_variable (bool): Whether or not between-regime rates are allowed to vary
        parallel (bool): Whether to run in parallel
        ncores (int): If parallel is True, number of cores to run on
        out_file (str): Optional output file name. If given, results will written
          into the current directory with this string as
          the first part of the filename

    """
    nchar = len(set(chars))
    if nchar != 2:
        raise ValueError,"Binary characters only. Number of states given:{}".format(nchar)
    regime_combinations = make_regime_type_combos(nregime, nparams)
    regime_combinations = remove_redundant_regimes(regime_combinations)
    pars = [None] * len(regime_combinations)
    Qs = [np.zeros([4,4])] * len(regime_combinations)
    liks = [None] * len(regime_combinations)
    ncomb = len(regime_combinations)
    ar = create_hrm_ar(tree, chars, nregime)
    print "Testing {} regimes".format(ncomb)

    if parallel:
        pool = mp.Pool(processes = ncores)
        func = partial(optimize_regime, tree=tree, chars=chars, nregime=nregime,nparams=nparams,pi=pi, br_variable=br_variable, ar=ar, out_file=out_file)
        results = pool.map(func, regime_combinations)
        out = results
        return {r[0]:(r[1],r[2]) for r in out}

    else:
        for i, comb in enumerate(regime_combinations):
            mk_func = hrm_disctinct_regimes_likelihoodfunc(tree, chars, comb, pi=pi,
                                            findmin = True, br_variable=br_variable, ar=ar)
            if not br_variable:
                nfreeparams = nparams+1
            else:
                nfreeparams = nparams + (nregime**2 - nregime)*2
            x0 = [0.1] * nfreeparams
            opt = nlopt.opt(nlopt.LN_SBPLX, nfreeparams)
            opt.set_min_objective(mk_func)
            opt.set_lower_bounds(0)
            pars[i] = opt.optimize(x0)
            liks[i] = -mk_func(pars[i])
            Qs[i] = fill_distinct_regime_Q(comb, np.insert(pars[i],0,1e-15), nregime, nchar, br_variable=br_variable)
            if out_file is not None:
                with open(out_file+str(comb)+".p", "wb") as f:
                    pickle.dump((comb, liks[i], Qs[i]), f)
            print "{}/{} regimes tested".format(i+1,ncomb)
        return {r:(liks[i], Qs[i]) for i,r in enumerate(regime_combinations)}


def fill_Q_layout(regimetype, Qparams):
    """
    Fill a single regime matrix based on the regime type and the given
    parameters.

    """
    Q = np.array([[0,Qparams[regimetype[0]]],
                 [Qparams[regimetype[1]],0]])
    return Q


def hrm_disctinct_regimes_likelihoodfunc(tree, chars, regimetypes, pi="Equal", findmin=True, br_variable=False, ar=None):
    nregime = len(regimetypes)
    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))
    nregimeshift = (nregime**2 - nregime)*2
    if not br_variable:
        nregimeshift = 1
    if ar is None:
        var = create_hrm_ar(tree, chars, nregime, findmin)
    else:
        var = ar.copy()
    var["Q_layout"] = np.zeros([nregime,nobschar,nobschar])

    def likelihood_function(Qparams, grad=None):
        """
        NLOPT inputs the parameter array as well as a gradient object.
        The gradient object is ignored for this optimizer (LN_SBPLX)
        """
        if any(np.isnan(Qparams)):
            return var["nullval"]
        # Enforcing upper bound on parameters
        if (sum(Qparams) > (var["upperbound"]*2)) or any(Qparams <= 0):
            return var["nullval"]
        if any(sorted(Qparams[:-nregimeshift])!=Qparams[:-nregimeshift]):
            return var["nullval"]

        if any(Qparams[-nregimeshift:]>Qparams[-(nregimeshift+1)]):
            return var["nullval"]

        Qparams = np.insert(Qparams, 0, 1e-15)

        fill_distinct_regime_Q(regimetypes,Qparams, nregime,nobschar,var["Q"], var["Q_layout"], br_variable)

        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if findmin:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_mk(tree, chars, var["Q"],nregime, pi = pi, ar=var)
            if not np.isnan(logli):
                return x * logli# Minimizing negative log-likelihood

        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function


def fill_distinct_regime_Q(regimetypes, Qparams,nregime, nobschar, Q = None, Q_layout = None, br_variable=False):
    returnQ = False
    if Q is None:
        returnQ = True
        Q = np.zeros([nobschar*nregime,nobschar*nregime])
        Q_layout = np.zeros([nregime,nobschar,nobschar])
    for i,rtype in enumerate(regimetypes):
        Q_layout[i] = fill_Q_layout(rtype, Qparams)
    # Filling regime sub-Qs within Q matrix:
    for i,regime in enumerate(Q_layout):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        Q[subQ, subQ] = regime
    # Filling in between-regime values
    for i,submatrix_index in enumerate(itertools.permutations(range(nregime),2)):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        if not br_variable:
            np.fill_diagonal(Q[my_slice0,my_slice1],Qparams[-1])
        else:
            nregimeswitch =(nregime**2 - nregime)*2
            np.fill_diagonal(Q[my_slice0,my_slice1],Qparams[(-nregimeswitch+i*2):])
    np.fill_diagonal(Q, -np.sum(Q,1))
    if returnQ:
        return Q

def fit_hrm_model(tree, chars, nregime, mod, pi="Equal", findmin=True, initialvals = None):
    """
    Fit parameters for a single model specified by the user
    """
    nobschar = len(set(chars))
    if nobschar != 2:
        raise ValueError,"Binary characters only. Number of states given:{}".format(nchar)
    nchar = nobschar * nregime
    ar = create_hrm_ar(tree, chars, nregime)
    nfreeparams = len(set([i for i in mod if i != 0]))
    n_wr = nobschar**2 - nobschar
    mod_format = [tuple(mod[i:i+n_wr]) for i in range(0, n_wr*nregime, n_wr)]
    mod_format.append(tuple(mod[n_wr*nregime:]))

    mk_func = fit_hrm_model_likelihood(tree, chars, nregime, mod_format, pi, findmin)
    if initialvals is None:
        x0 = [0.1]*nfreeparams
    else:
        x0 = initialvals
    opt = nlopt.opt(nlopt.LN_SBPLX, nfreeparams)
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)

    par = opt.optimize(x0)
    lik = -mk_func(par)
    Q = np.zeros([nchar, nchar])
    fill_model_Q(mod_format, np.insert(par, 0, 1e-15), Q)

    return Q, lik

def fill_model_Q(mod, Qparams, Q):
    """
    Fill Q matrix given model and parameters.

    Args:
        mod (list): List of tuples. Code for the mode used. The first nregime
          tuples correspond to the within-regime transition rates for each
          regime. The last tuple corresponds to the between-regime transition
          rates. Ex. [(1, 2), (3, 4), (5, 6, 7, 8)] corresponds to a
          matrix of:
              [-,1,5,-]
              [2,-,-,6]
              [7,-,-,3]
              [-,8,4,-]
        Qparams (list): List of floats corresponding to the values for
          slow, medium, fast, etc. rates. The first is always 0
        Q (np.array): Pre-allocated Q matrix
    """
    nregime = len(mod)-1
    nobschar = Q.shape[0]/nregime
    nchar = Q.shape[0]
    Q.fill(0.0)
    for i in range(nregime):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        subQvals = [Qparams[x] for s in [(0,)+mod[i][k:k+nobschar] for k in range(0,len(mod[i])+1,nobschar)] for x in s]
        np.copyto(Q[subQ, subQ], [subQvals[x:x+nobschar] for x in xrange(0, len(subQvals), nobschar)])

    combs = list(itertools.combinations(range(nregime),2))
    revcombs = [tuple(reversed(i)) for i in combs]
    submatrix_indices = [x for s in [[combs[i]] + [revcombs[i]] for i in range(len(combs))] for x in s]
    for i,submatrix_index in enumerate(submatrix_indices):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        nregimeswitch =(nregime**2 - nregime)*2
        np.fill_diagonal(Q[my_slice0,my_slice1],[Qparams[p] for p in mod[nregime][i*nobschar:i*nobschar+nobschar]])
    np.fill_diagonal(Q, -np.sum(Q,1))


def fit_hrm_model_likelihood(tree, chars, nregime, mod, pi="Equal", findmin=True):
    """
    Likelihood function for fitting user-specified model
    """
    nobschar = len(set(chars))
    nchar = nobschar*nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)

    var = create_hrm_ar(tree, chars, nregime, findmin)
    var["Q_layout"] = np.zeros([nregime,nobschar,nobschar])
    def likelihood_function(Qparams, grad=None):
        """
        NLOPT inputs the parameter array as well as a gradient object.
        The gradient object is ignored for this optimizer (LN_SBPLX)
        """
        if any(np.isnan(Qparams)):
            return var["nullval"]
        if not all(sorted(Qparams) == Qparams):
            return var["nullval"]

        if (sum(Qparams) > (var["upperbound"]*2)) or any(Qparams <= 0):
            return var["nullval"]


        Qparams = np.insert(Qparams, 0, 1e-15)

        fill_model_Q(mod, Qparams, var["Q"])

        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if findmin:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_mk(tree, chars, var["Q"],nregime, pi = pi, ar=var)
            if not np.isnan(logli):
                return x * logli# Minimizing negative log-likelihood

        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function


def cluster_models(tree, chars, Q, nregime, pi="Equal", findmin=True):
    """
    Given an MLE Q, return candidate models with more parsimonious
    parameter set by merging similar parameters
    """
    Qparams = extract_Qparams(Q, nregime)

    ts = np.linspace(-10, 0, 11)
    Q_dist = np.array(zip(list(Qparams), [0]*len(Qparams)))
    candidate_models = list(set([tuple(scipy.cluster.hierarchy.fclusterdata(Q_dist, i)) for i in ts]))
    if any(np.isclose(Qparams, 0.0)):
        for i,c in enumerate(candidate_models):
            candidate_models[i] = tuple([x-1 for x in c])

    nmod = len(candidate_models)
    print("Testing {} models".format(nmod))
    alt_mods = {c:fit_hrm_model(tree,chars,nregime,c,pi=pi,findmin=findmin)
                for c in candidate_models}
    return alt_mods


def AIC(l, k):
    return 2*k - 2*l


def pairwise_param_merging(tree, chars, Q, nregime, pi="Equal", findmin=True):
    """
    Given an MLE Q, merge similar parameters pairwise to find more
    parsimonious models
    """
    Qparams = extract_Qparams(Q, nregime)
    prev_mod = Qparams.argsort().argsort() + 1
    prev_l = fit_hrm_model(tree,chars,nregime,tuple(prev_mod),pi=pi,findmin=findmin,
                           initialvals=sorted(Qparams))[1]
    prev_AIC = AIC(prev_l, len(Qparams))
    prev_Q = Q

    dist_mat = abs(Qparams[..., np.newaxis] - Qparams[np.newaxis, ...])
    dist_mat[np.tril_indices(len(Qparams))] = np.inf

    while 1:
        closest_pair = divmod(np.argmin(dist_mat), len(Qparams))
        greater = max(prev_mod[list(closest_pair)])
        new_mod = np.array([i if i<greater else i-1 for i in prev_mod])

        initialvals = sorted(list(set(Qparams[new_mod])))

        new_Q, new_l = fit_hrm_model(tree,chars,nregime,tuple(new_mod),pi=pi,findmin=findmin,
                                     initialvals=initialvals)
        new_AIC = AIC(new_l, len(set(new_mod)))
        if new_AIC < prev_AIC:
            # dist_mat[closest_pair[0], closest_pair[1]] = np.inf
            dist_mat[closest_pair[1],] = np.inf
            dist_mat[:,closest_pair[1]] = np.inf
            prev_mod = new_mod[:]
            prev_AIC = new_AIC
            prev_l = new_l
            prev_Q = new_Q
        else:
            best_mod = prev_mod
            best_Q = prev_Q
            best_l = prev_l
            break
    return [best_mod, best_Q, best_l]


def extract_Qparams(Q, nregime):
    nchar = Q.shape[0]
    nobschar = nchar/nregime

    n_wr = (nobschar**2-nobschar)
    n_br = (nregime**2-nregime)*nobschar

    Qparams = np.zeros([n_wr*nregime + n_br])

    for i in range(nregime):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        mask = np.ones([nobschar,nobschar], dtype=bool)
        mask[np.diag_indices(nobschar)]=False
        np.copyto(Qparams[i*n_wr:(i+1)*n_wr], Q[subQ,subQ][mask])

    combs = list(itertools.combinations(range(nregime),2))
    revcombs = [tuple(reversed(i)) for i in combs]
    submatrix_indices = [x for s in [[combs[i]] + [revcombs[i]] for i in range(len(combs))] for x in s]
    for i,submatrix_index in enumerate(submatrix_indices):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        nregimeswitch =(nregime**2 - nregime)*2
        Qparams[n_wr*nregime + i*nobschar:n_wr*nregime+(i+1)*nobschar] = Q[my_slice0,my_slice1][np.diag_indices(nobschar)]
    return Qparams
