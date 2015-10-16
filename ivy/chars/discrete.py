"""
Functions for fitting mk model to a tree
"""
import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom

def nodeLikelihood(node):
    """
    Take node "node" and calculate its likelihood given its children's likelihoods,
    branch lengths, and p-matrix.
    Args:
        node (Node): A node to calculate the likelihood for
    Returns:
        float: The likelihood of the node given the data
    """
    likelihoodNode = {}
    for state in range(node.children[0].pmat.shape[0]): # Calculate the likelihood of the node being any one of these states
        likelihoodStateN = [] # Likelihood of node being at state N
        for ch in node.children:
            likelihoodStateN.append(ch.pmat[state, ch.charstate])
        likelihoodNode[state] = np.product(likelihoodStateN)

    return sum(likelihoodNode.values())

def mk(tree, chars, Q, p = None, pi="Equal"):
    """
    Fit mk model and return likelihood for the root node of a tree given a list of characters
    and a Q matrix

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        Q (np.array): Instantaneous rate matrix
        p (np.array): Optional pre-allocated p matrix
        pi (str or np.array): Option to weight the root node by given values.
                       If None, defaults to equal weights. Weights should
                       be given in order.
    """
    chartree = tree.copy()
    chartree.char = None; chartree.likelihoodNode={}
    t = [node.length for node in chartree.descendants()]
    t = np.array(t, dtype=np.double)

    if p is None:
        p = np.empty([len(t), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    cyexpokit.dexpm_tree_preallocated_p(Q, t, p) # This changes p in place

    for i, nd in enumerate(chartree.descendants()):
        nd.pmat = p[i]
        nd.likelihoodNode = {}
        nd.char = None

    for i, lf in enumerate(chartree.leaves()):
        lf.char = chars[i]


    for node in chartree.postiter():
        if node.char is not None: # For tip nodes, likelihoods are 1 for observed state and 0 for the rest
            for state in range(Q.shape[0]):
                node.likelihoodNode[state]=0.0
            node.likelihoodNode[node.char]=1.0
        else:
            for state in range(Q.shape[0]):
                likelihoodStateN = []
                for ch in node.children:
                    likelihoodStateNCh = []
                    for chState in range(Q.shape[0]):
                        likelihoodStateNCh.append(ch.pmat[state, chState] * ch.likelihoodNode[chState]) #Likelihood for a certain state = p(stateBegin, stateEnd * likelihood(stateEnd))
                    likelihoodStateN.append(sum(likelihoodStateNCh))
                node.likelihoodNode[state]=np.product(likelihoodStateN)
    nchar = len(chartree.likelihoodNode.values())

    if type(pi) != str:
        li = sum([ i*pi[n] for n,i in chartree.likelihoodNode.iteritems() ])
        return(math.log(li))

    elif pi == "Equal":
        li = sum([ i/nchar for i in chartree.likelihoodNode.values() ])
        return(math.log(li)) # Assuming a flat pi: multiply each likelihood by prior (1/nchar)

    elif pi == "Fitzjohn":
        rootPriors = { charstate:chartree.likelihoodNode[charstate]/
                                 sum(chartree.likelihoodNode.values()) for
                                 charstate in set(chars) }

        li = sum([ chartree.likelihoodNode[charstate] *
                     rootPriors[charstate] for charstate in set(chars) ])
        return(math.log(li))
    elif pi == "Equilibrium":

        rootPriors = qsd(Q)

        li = sum([ i*rootPriors[n] for n,i in chartree.likelihoodNode.iteritems() ])
        return(math.log(li))

def create_likelihood_function_ER(tree, chars, pi="Equal"):
    """
    Create a function that takes values for an ER Q and returns likelihood.

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node.
    Returns:
        function: Function accepting a list of length 1 as its only
          argument and returning log-likelihood
    """
    nchar = len(set(chars))

    # Empty Q matrix
    Q = np.identity(nchar, dtype=np.double)
    nt = len(tree.descendants())
    p = np.empty([nt, Q.shape[0], Q.shape[1]], dtype = np.double, order="C")

    var = {"Q": Q, "p": p} # Giving internal function access to these arrays.
                           # Warning: can be tricky - Q and p are initialized ONCE,
                           # Then their mutated values are re-used.
                           # Need to make sure old values
                           # Aren't accidentally re-used

    def likelihood_function(Qparam):
        nchar = len(set(chars))

        param = Qparam[0]

        # Fill in Q matrix
        var["Q"].fill(param)
        var["Q"][np.diag_indices(nchar)] = -param * (nchar-1)

        try:
            return -1 * mk(tree, chars, var["Q"], var["p"], pi) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return 1e255

    return likelihood_function

def create_likelihood_function_Sym(tree, chars, pi="Equal"):
    """
    Create a function that takes values for an symmetrical Q and returns likelihood.

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node.
    Returns:
        function: Function accepting a list of parameters
          and returning log-likelihood
    """
    nchar = len(set(chars))

    Q = np.zeros([nchar,nchar], dtype=np.double)
    nt = len(tree.descendants())
    p = np.empty([nt, Q.shape[0], Q.shape[1]], dtype = np.double, order="C")

    var = {"Q":Q, "p":p} # Giving internal function access to these arrays.
                         # Warning: can be tricky - Q and p are initialized ONCE,
                         # Then their mutated values are re-used.
                         # Need to make old values aren't
                         # Accidentally re-used

    def likelihood_function(Qparams):
        """
        Number of params equal to binom(nchar, 2). Forward and backwards
        transitions are the same
        """

        var["Q"].fill(0.0) # Re-filling with zeroes

        nchar = len(set(chars))
        params = Qparams[:]
        # Generating symmetric Q matrix

        xs,ys = np.triu_indices(nchar,k=1)
        var["Q"][xs,ys] = params
        var["Q"][ys,xs] = params
        var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)

        try:
            return -1 * mk(tree, chars, var["Q"], p=var["p"], pi=pi) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return 1e255

    return likelihood_function

def create_likelihood_function_ARD(tree, chars, pi="Equal"):
    """
    Create a function that takes values for an ARD Q and returns likelihood.

    Returned function to be passed into scipy.optimize

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be numbered 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node.
    Returns:
        function: Function accepting a list of parameters
          and returning log-likelihood
    """
    nchar = len(set(chars))

    Q = np.zeros([nchar,nchar], dtype=np.double) # Generating empty matrix
    nt = len(tree.descendants())
    p = np.empty([nt, Q.shape[0], Q.shape[1]], dtype = np.double, order="C")

    var = {"Q": Q, "p": p} # Giving internal function access to these arrays.
                           # Warning: can be tricky - Q and p are initialized ONCE,
                           # Then their mutated values are re-used.
                           # Need to make sure old values
                           # Aren't accidentally re-used
    def likelihood_function(Qparams):
        """
        Number of params equal to binom(nchar, 2). Forward and backwards
        transitions are the same
        """
        var["Q"].fill(0.0) # Re-filling with zeroes

        nchar = len(set(chars))
        params = Qparams[:]
        # Filling in values for ARD matrix

        var["Q"][np.triu_indices(nchar, k=1)] = params[:len(params)/2]
        var["Q"][np.tril_indices(nchar, k=-1)] = params[len(params)/2:]
        var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)

        try:
            return -1 * mk(tree, chars, var["Q"], p=var["p"], pi = pi) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return 1e255

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
        tuple: Fitted parameter and log-likelihood

    """
    nchar = len(set(chars))
    # Initial value arbitrary
    x0 = [0.1] # Starting value for our equal rates model
    mk_func = create_likelihood_function_ER(tree, chars, pi)

    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = [(0,None)])

    q = np.empty([nchar,nchar], dtype=np.double)
    q.fill(optim.x[0])

    q[np.diag_indices(nchar)] = 0 - (q.sum(1)-q[0,0])

    return (q, -1*float(optim.fun))

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
        tuple: Tuple of fitted parameters and log-likelihood

    """

    nchar = len(set(chars))
    # Number of params equal to binom(nchar, 2)
    # Initial values arbitrary
    x0 = [0.1] * binom(nchar, 2) # Starting values for our symmetrical rates model
    mk_func = create_likelihood_function_Sym(tree, chars, pi = pi)

    # Need to constrain values to be greater than 0
    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = tuple(( (0,None) for i in range(len(x0)) )))


    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x
    q = q + q.T
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)



    return (q, -1*float(optim.fun))


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
        tuple: Tuple of fitted parameters and log-likelihood

    """
    # Number of parameters equal to k^2 - k
    nchar = len(set(chars))
    x0 = [1.0] * (nchar ** 2 - nchar)

    mk_func = create_likelihood_function_ARD(tree, chars, pi=pi)

    optim = minimize(mk_func, x0, method="L-BFGS-B",
                      bounds = tuple(( (0,None) for i in range(len(x0)) )))

    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x[:len(optim.x)/2]
    q[np.tril_indices(nchar, k=-1)] = optim.x[len(optim.x)/2:]
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    return (q, -1*float(optim.fun))


def fitMk(tree, chars, Q = "Equal", pi = "Equal"):
    """
    Fit an mk model to a given tree and list of characters. Return fitted
    Q matrix and calculated likelihood.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        pi (str): Either "Equal" or "Fitzjohn". How to weight values at root
          node. Defaults to "Equal"
          Method "Fitzjohn" is currently untested
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
            q,l = fitMkER(tree, chars, pi="Equal")
            if pi == "Equal":
                return q,l

            # For likelihood calculation of non-flat pi,
            # Calculate Q under flat pi, then use the pi from that Q
            # to determine likelihood
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l = fitMkER(tree, chars, pi=pivals)
                return q,l
            elif pi =="Fitzjohn":
                pass
        elif Q == "Sym":
            q,l = fitMkSym(tree, chars, pi="Equal")
            if pi == "Equal":
                return q,l
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l = fitMkSym(tree, chars, pi=pivals)
                return q,l
            elif pi =="Fitzjohn":
                pass
        elif Q == "ARD":
            q,l = fitMkARD(tree, chars, pi="Equal")
            if pi == "Equal":
                return q,l
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l = fitMkARD(tree, chars, pi=pivals)
                return q,l
            elif pi =="Fitzjohn":
                pass
        else:
            raise ValueError("Q str must be one of: 'Equal', 'Sym', 'ARD'")

    else:
        assert str(type(Q)) == "<type 'numpy.ndarray'>", "Q must be str or numpy array"
        assert len(Q[0]) == len(set(chars)), "Supplied Q has wrong dimensions"

        loglik = mk(tree, chars, Q, pi="Equal")

        return Q,loglik

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
            constraints = {"type":"eq",
                           "fun": lambda x: 1 - sum(x)},
            method = "SLSQP",
            options = {"ftol":1e-14})

    return optim.x
