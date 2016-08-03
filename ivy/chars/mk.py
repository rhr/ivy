from __future__ import absolute_import, division, print_function, unicode_literals

import math
import random
import types

import numpy as np
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
import nlopt
try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]

def mk(tree, chars, Q, p=None, pi="Equal",returnPi=False, ar=None):
    """
    Fit mk model and return likelihood for the root node of a tree given a list of characters
    and a Q matrix

    Convert tree and character data into a form that can be input
    into mk, which fits an mk model.

    Note: internal calculations are log-transformed to avoid numeric underflow

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
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
    Returns:
        float: log-likelihood of model
    Examples:
        from ivy.examples import primates, primate_data
        Q = np.array([[-0.1,0.1],[0.1,-0.1]])
        mk(primates,primate_data,Q)

    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = Q.shape[0]
    if ar is None:
        # Creating arrays to be used later
        ar = create_mk_ar(tree, chars)
    cyexpokit.dexpm_tree_preallocated_p_log(Q, ar["t"], ar["p"]) # This changes p in place
    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"],ar["intnode_list"],
                        ar["child_ar"])
    # The last row of nodelist contains the likelihood values at the root

    # Applying the correct root prior
    if not type(pi) in StringTypes:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) in ["<type 'numpy.ndarray'>","<class 'numpy.ndarray'>"], "pi must be str or numpy array"
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
                   charstate in set(chars) ])
        rootliks = [ ar["nodelist"][-1,:-1][charstate] +
                     ar["root_priors"][charstate] for charstate in set(chars) ]
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(ar["root_priors"],qsd(Q))
        rootliks = [ i + np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ]
    logli = scipy.misc.logsumexp(rootliks)
    if returnPi:
        return (logli, {k:v for k,v in enumerate(ar["root_priors"])}, rootliks)
    else:
        return logli


def create_mk_ar(tree, chars, findmin = True):
    """
    Create preallocated arrays. For use in mk function

    Nodelist = edgelist of nodes in postorder sequence
    """
    for n in tree:
        n.cladesize = len(n)
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nt = len(tree.descendants())
    nchar = len(set(chars))
    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    nodelist = np.zeros((nnode, nchar+1))
    nodelist.fill(-np.inf) # the log of 0 is negative infinity
    leafind = [ n.isleaf for n in tree.postiter()]

    for k,ch in enumerate(postChars):
        [ n for i,n in enumerate(nodelist) if leafind[i] ][k][ch] = np.log(1.0)
    for i,n in enumerate(nodelist[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)

    # Setting initial node likelihoods to log one for calculations
    nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = np.log(1.0)

    # Empty Q matrix
    Q = np.zeros([nchar, nchar], dtype=np.double)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    nodelistOrig = nodelist.copy()
    rootpriors = np.empty([nchar], dtype=np.double)
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf
    treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
                   tree.leaves()[0].length])
    upperbound = len(tree.leaves())/treelen
    charlist = list(range(nchar))
    tmp_ar = np.zeros(nchar)

    max_children = max(len(n.children) for n in tree)
    child_ar = np.empty([tree.cladesize,max_children], dtype=np.int64) # List of children per node for cython function
    child_ar.fill(-1)

    intnode_list = np.array(sorted(set(nodelist[:-1,nchar])),dtype=int) # Internal node list for cython function
    for intnode in intnode_list:
        children = np.where(nodelist[:,nchar]==intnode)[0]
        child_ar[int(intnode)][:len(children)] = children

    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used
    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar,
           "intnode_list":intnode_list, "child_ar":child_ar}
    return var


def create_likelihood_function_mk(tree, chars, Qtype, pi="Equal",
                                  findmin = True):
    """
    Create a function that takes values for Q and returns likelihood.

    Specify the Q to be ER, Sym, or ARD

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        Qtype (str): What type of Q matrix to use. Either ER (equal rates),
          Sym (symmetric rates), or ARD (All rates different).
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root  node.
        min (bool): Whether the function is to be minimized (False means
          it will be maximized)
    Returns:
        function: Function accepting a list of parameters and returning
          log-likelihood. To be optmimized with NLOPT
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf
    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = list(range(nchar))
    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used
    var = create_mk_ar(tree, chars, findmin)
    def likelihood_function(Qparams, grad=None):
        """
        NLOPT supplies likelihood function with parameters and gradient
        """
        # Enforcing upper bound on parameters
        if (sum(Qparams) > var["upperbound"]) or any(Qparams <= 0):
            return var["nullval"]
        if any(np.isnan(Qparams)):
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
            var["Q"][np.triu_indices(nchar, k=1)] = Qparams[:int(len(Qparams)/2)]
            var["Q"][np.tril_indices(nchar, k=-1)] = Qparams[int(len(Qparams)/2):]
            var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)
        else:
            raise ValueError("Qtype must be one of: ER, Sym, ARD")
        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)
        if findmin:
            x = -1
        else:
            x = 1
        try:
            li = mk(tree, chars, var["Q"], p=var["p"], pi = pi, ar=var)
            return x * li # Minimizing negative log-likelihood
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
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.

    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = len(set(chars))
    # Initial value arbitrary
    x0 = [.5] # Starting value for our equal rates model
    mk_func = create_likelihood_function_mk(tree, chars, Qtype="ER", pi=pi)

    opt = nlopt.opt(nlopt.LN_SBPLX, len(x0))
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)

    optim = opt.optimize(x0)

    q = np.empty([nchar,nchar], dtype=np.double)
    q.fill(optim[0])

    q[np.diag_indices(nchar)] = 0 - (q.sum(1)-q[0,0])

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*mk_func(optim), piRates, rootLiks)

def fitMkSym(tree, chars, pi="Equal"):
    """
    Estimate parameter of a symmetrical-rate Q matrix
    Return log-likelihood of mk equation using fitted Q

    Multi-parameter model: forward = reverse

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"
          Method "Fitzjohn" is currently untested

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.


    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = len(set(chars))
    # Number of params equal to binom(nchar, 2)
    # Initial values arbitrary
    x0 = [0.5] * int(binom(nchar, 2)) # Starting values for our symmetrical rates model
    mk_func = create_likelihood_function_mk(tree, chars, Qtype="Sym", pi = pi)

    # Need to constrain values to be greater than 0
    opt = nlopt.opt(nlopt.LN_SBPLX, len(x0))
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)

    optim = opt.optimize(x0)

    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim
    q = q + q.T
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*mk_func(optim), piRates, rootLiks)


def fitMkARD(tree, chars, pi="Equal"):
    """
    Estimate parameters of an all-rates-different Q matrix
    Return log-likelihood of mk equation using fitted Q

    Multi-parameter model: all rates different

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"
          Method "Fitzjohn" is currently untested

    Returns:
        tuple: Fitted parameter, log-likelihood, and dictionary of weightings
          at the root.

    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    # Number of parameters equal to k^2 - k
    nchar = len(set(chars))
    x0 = [.5] * (nchar ** 2 - nchar)

    mk_func = create_likelihood_function_mk(tree, chars, Qtype="ARD", pi=pi)

    opt = nlopt.opt(nlopt.LN_SBPLX, len(x0))
    opt.set_min_objective(mk_func)
    opt.set_lower_bounds(0)

    optim = opt.optimize(x0)

    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim[:int(len(optim)/2)]
    q[np.tril_indices(nchar, k=-1)] = optim[int(len(optim)/2):]
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates, rootLiks = mk(tree, chars, q, pi=pi, returnPi=True)[1:]

    return (q, -1*mk_func(optim), piRates, rootLiks)


def fit_Mk(tree, chars, Q = "Equal", pi = "Equal"):
    """
    Fit an mk model to a given tree and list of characters. Return fitted
    Q matrix and calculated likelihood.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
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
        dict: Log-likelihood, fitted Q matrix, root prior, root likelihood

    Examples:
        from ivy.examples import primates, primate_data
        primate_eq = fit_Mk(primates,primate_data,Q="Equal")
        primate_ARD = fit_Mk(primates, primate_data,Q="ARD")
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    assert pi in ["Equal", "Fitzjohn", "Equilibrium"], "Pi must be one of: 'Equal', 'Fitzjohn', 'Equilibrium'"

    if type(Q) in StringTypes:
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
        assert str(type(Q)) in ["<type 'numpy.ndarray'>" ,"<class 'numpy.ndarray'>"], "Q must be str or numpy array"
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
