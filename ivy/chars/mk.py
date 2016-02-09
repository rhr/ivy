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

def mk(tree, chars, Q, p=None, pi="Equal",returnPi=False,
          preallocated_arrays=None):
    """
    Fit mk model and return likelihood for the root node of a tree given a list of characters
    and a Q matrix

    Convert tree and character data into a form that can be input
    into mk, which fits an mk model.

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
        preallocated_arrays (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays
    """
    nchar = Q.shape[0]
    if preallocated_arrays is None:
        # Creating arrays to be used later
        preallocated_arrays = {}
        preallocated_arrays["t"] = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
        preallocated_arrays["charlist"] = range(Q.shape[0])
    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place

    if len(preallocated_arrays.keys())==2:
        # Creating more arrays
        nnode = len(tree)
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

        rootliks = [ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ]

    elif pi == "Equal":
        preallocated_arrays["root_priors"].fill(1.0/nchar)
        rootliks = [ float(i)/nchar for i in preallocated_arrays["nodelist"][-1] ][:-1]

    elif pi == "Fitzjohn":
        np.copyto(preallocated_arrays["root_priors"],
                  [preallocated_arrays["nodelist"][-1,:-1][charstate]/
                   sum(preallocated_arrays["nodelist"][-1,:-1]) for
                   charstate in set(chars) ])

        rootliks = [ preallocated_arrays["nodelist"][-1,:-1][charstate] *
                     preallocated_arrays["root_priors"][charstate] for charstate in set(chars) ]

    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(preallocated_arrays["root_priors"],qsd(Q))
        rootliks = [ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ]
    li = sum(rootliks)
    logli = math.log(li)
    if returnPi:
        return (logli, {k:v for k,v in enumerate(preallocated_arrays["root_priors"])}, rootliks)
    else:
        return logli



# Pure python mk function
# def mk_py(tree, chars, Q, p = None, pi="Equal", returnPi=False):
#     """
#     Fit mk model and return likelihood for the root node of a tree given a list of characters
#     and a Q matrix
#
#     Args:
#         tree (Node): Root node of a tree. All branch lengths must be
#           greater than 0 (except root)
#         chars (list): List of character states corresponding to leaf nodes in
#           preoder sequence. Character states must be numbered 0,1,2,...
#         Q (np.array): Instantaneous rate matrix
#         p (np.array): Optional pre-allocated p matrix
#         pi (str or np.array): Option to weight the root node by given values.
#                        Either a string containing the method or an array
#                        of weights. Weights should be given in order.
#
#                        Accepted methods of weighting root:
#
#                        Equal: flat prior
#                        Equilibrium: Prior equal to stationary distribution
#                          of Q matrix
#                        Fitzjohn: Root states weighted by how well they
#                          explain the data at the tips.
#         returnPi (bool): Whether or not to return the pi used
#     """
#     chartree = tree.copy()
#     chartree.char = None; chartree.likelihoodNode={}
#     t = [node.length for node in chartree.descendants()]
#     t = np.array(t, dtype=np.double)
#     nchar = Q.shape[0]
#
#     # Generating probability matrix for each branch
#     if p is None:
#         p = np.empty([len(t), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
#     cyexpokit.dexpm_tree_preallocated_p(Q, t, p) # This changes p in place
#
#
#     for i, nd in enumerate(chartree.descendants()):
#         nd.pmat = p[i] # Assigning probability matrices for each branch
#         nd.likelihoodNode = {}
#         nd.char = None
#
#     for i, lf in enumerate(chartree.leaves()):
#         lf.char = chars[i] # Assigning character states to tips
#
#
#     for node in chartree.postiter():
#         if node.char is not None: # For tip nodes, likelihoods are 1 for observed state and 0 for the rest
#             for state in range(nchar):
#                 node.likelihoodNode[state]=0.0
#             node.likelihoodNode[node.char]=1.0
#         else:
#             for state in range(nchar):
#                 likelihoodStateN = []
#                 for ch in node.children:
#                     likelihoodStateNCh = []
#                     for chState in range(nchar):
#                         likelihoodStateNCh.append(ch.pmat[state, chState] * ch.likelihoodNode[chState]) #Likelihood for a certain state = p(stateBegin, stateEnd * likelihood(stateEnd))
#                     likelihoodStateN.append(sum(likelihoodStateNCh))
#                 node.likelihoodNode[state]=np.product(likelihoodStateN)
#
#     if type(pi) != str:
#         assert len(pi) == nchar, "length of given pi does not match Q dimensions"
#         assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
#         assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"
#
#
#         rootPriors = {state:li for state,li in enumerate(pi)}
#
#         li = sum([ i*rootPriors[n] for n,i in chartree.likelihoodNode.iteritems() ])
#         logli = math.log(li)
#
#     elif pi == "Equal":
#         rootPriors = {state:1.0/nchar for state in range(nchar)}
#         li = sum([ i/nchar for i in chartree.likelihoodNode.values() ])
#
#         logli = math.log(li)
#
#     elif pi == "Fitzjohn":
#         rootPriors = { charstate:chartree.likelihoodNode[charstate]/
#                                  sum(chartree.likelihoodNode.values()) for
#                                  charstate in set(chars) }
#
#         li = sum([ chartree.likelihoodNode[charstate] *
#                      rootPriors[charstate] for charstate in set(chars) ])
#         logli = math.log(li)
#     elif pi == "Equilibrium":
#         # Equilibrium pi from the stationary distribution of Q
#         rootPriors = {state:li for state,li in enumerate(qsd(Q))}
#
#         li = sum([ i*rootPriors[n] for n,i in chartree.likelihoodNode.iteritems() ])
#         logli = math.log(li)
#     if returnPi:
#         return (logli, rootPriors)
#     else:
#         return logli

def _create_nodelist(tree, chars):
    """
    Create nodelist. For use in mk function

    Returns edgelist of nodes in postorder sequence
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

def create_likelihood_function_mk(tree, chars, Qtype, pi="Equal",
                                  min = True):
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
        min (bool): Whether the function is to be minimized (False means
          it will be maximized)
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
        if (sum(Qparams) > var["upperbound"]) or any(Qparams <= 0):
            return var["nullval"]

        # Filling Q matrices:
        if Qtype == "ER":
            var["Q"].fill(Qparams[0])
            var["Q"][np.diag_indices(nchar)] = -Qparams[0] * (nchar-1)
        elif Qtype == "Sym":
            var["Q"].fill(0.0) # Re-filling with zeroes
            xs,ys = np.triu_indices(nchar,k=1)
            var["Q"][xs,ys] = Qparams
            var["Q"][ys,xs] = Qparams
            var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        elif Qtype == "ARD":
            var["Q"].fill(0.0) # Re-filling with zeroes
            var["Q"][np.triu_indices(nchar, k=1)] = Qparams[:len(Qparams)/2]
            var["Q"][np.tril_indices(nchar, k=-1)] = Qparams[len(Qparams)/2:]
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
            return x * mk(tree, chars, var["Q"], p=var["p"], pi = pi, preallocated_arrays=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function


def fitMkER(tree, chars, pi="Equal"):
    """
    Estimate parameter of an equal-rate Q matrix
    Return log-likelihood of mk equation using fitted Q

    One-parameter model: alpha = beta

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.

    """
    nchar = len(set(chars))
    # Initial value arbitrary
    x0 = [0.1] # Starting value for our equal rates model
    mk_func = create_likelihood_function_mk(tree, chars, Qtype="ER", pi=pi)

    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = [(1e-14,None)])

    q = np.empty([nchar,nchar], dtype=np.double)
    q.fill(optim.x[0])

    q[np.diag_indices(nchar)] = 0 - (q.sum(1)-q[0,0])

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*float(optim.fun), piRates, rootLiks)

def fitMkSym(tree, chars, pi="Equal"):
    """
    Estimate parameter of a symmetrical-rate Q matrix
    Return log-likelihood of mk equation using fitted Q

    Multi-parameter model: forward = reverse

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"
          Method "Fitzjohn" is currently untested

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.


    """

    nchar = len(set(chars))
    # Number of params equal to binom(nchar, 2)
    # Initial values arbitrary
    x0 = [0.1] * binom(nchar, 2) # Starting values for our symmetrical rates model
    mk_func = create_likelihood_function_mk(tree, chars, Qtype="Sym", pi = pi)

    # Need to constrain values to be greater than 0
    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))


    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x
    q = q + q.T
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*float(optim.fun), piRates, rootLiks)


def fitMkARD(tree, chars, pi="Equal"):
    """
    Estimate parameters of an all-rates-different Q matrix
    Return log-likelihood of mk equation using fitted Q

    Multi-parameter model: all rates different

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"
          Method "Fitzjohn" is currently untested

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.

    """
    # Number of parameters equal to k^2 - k
    nchar = len(set(chars))
    x0 = [1.0] * (nchar ** 2 - nchar)

    mk_func = create_likelihood_function_mk(tree, chars, Qtype="ARD", pi=pi)

    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))

    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x[:len(optim.x)/2]
    q[np.tril_indices(nchar, k=-1)] = optim.x[len(optim.x)/2:]
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*float(optim.fun), piRates, rootLiks)


def fitMk(tree, chars, Q = "Equal", pi = "Equal"):
    """
    Fit an mk model to a given tree and list of characters. Return fitted
    Q matrix and calculated likelihood.

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

    Returns:
        tuple: Tuple of fitted Q matrix (a np array) and log-likelihood value
    """
    assert pi in ["Equal", "Fitzjohn", "Equilibrium"], "Pi must be one of: 'Equal', 'Fitzjohn', 'Equilibrium'"

    if type(Q) == str:
        if Q == "Equal":
            q,l,piRates,rootLiks = fitMkER(tree, chars, pi=pi)

        elif Q == "Sym":
            q,l,piRates,rootLiks = fitMkSym(tree, chars, pi=pi)

        elif Q == "ARD":
            q,l,piRates,rootLiks = fitMkARD(tree, chars, pi=pi)
        else:
            raise ValueError("Q str must be one of: 'Equal', 'Sym', 'ARD'")

        return {key:val for key, val in zip(["Q", "Log-likelihood","pi","rootLiks"], [q,l,piRates,rootLiks])}


    else:
        assert str(type(Q)) == "<type 'numpy.ndarray'>", "Q must be str or numpy array"
        assert len(Q[0]) == len(set(chars)), "Supplied Q has wrong dimensions"

        l,piRates, rootLiks = mk(tree, chars, Q, pi=pi, returnPi=True)
        q = Q

        return {key:val for key, val in zip(["Q", "Log-likelihood","pi","rootLiks"], [q,l,piRates,rootLiks])}

def qsd(Q):
    """
    Calculate stationary distribution of Q, assuming each state
    has the same diversification rate.

    Args:
        Q (np.array): Instantaneous rate matrix

    Returns:
        (np.array): Stationary distribution of pi

    Eqn from Maddison et al 2007

    Referenced from Phytools (Revell 2013)
    """
    nchar = Q.shape[0]
    def qsd_root(pi):
        return sum(np.dot(pi, Q)**2)
    x0 = [1.0/nchar]*nchar
    optim = minimize(qsd_root, x0,
            bounds = tuple(( (1e-14,None) for i in range(len(x0)) )),
            constraints = {"type":"eq",
                           "fun": lambda x: 1 - sum(x)},
            method = "SLSQP",
            options = {"ftol":1e-14})

    return optim.x
