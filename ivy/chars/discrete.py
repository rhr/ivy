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
def cy_mk(tree, chars, Q, p=None, nodelist=None, pi="Equal",returnPi=False,
          preallocated_arrays=None):
    """
    Fit mk model and return likelihood for the root node of a tree given a list of characters
    and a Q matrix

    Convert tree and character data into a form that can be input
    into cy_mk, which fits an mk model.

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
    """
    nchar = Q.shape[0]
    if preallocated_arrays is None:
        preallocated_arrays = {}

        t = [node.length for node in tree.postiter() if not node.isroot]
        t = np.array(t, dtype=np.double)

        preallocated_arrays["charlist"] = range(Q.shape[0])
        preallocated_arrays["t"] = t


    if p is None:
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place

    if len(preallocated_arrays.keys())==2:

        nnode = len(tree.descendants())+1

        preallocated_arrays["nodelist"] = np.zeros((nnode, nchar+1))
        leafind = [ n.isleaf for n in tree.postiter()]

        # Reordering character states to be in postorder sequence
        preleaves = [ n for n in tree.preiter() if n.isleaf ]
        postleaves = [n for n in tree.postiter() if n.isleaf ]
        postnodes = list(tree.postiter())
        prenodes = list(tree.preiter())
        postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]

        temp = [p[i] for i in [ postnodes.index(n) for n in prenodes if not n.isroot] ]
        temp2 = [t[i] for i in [ postnodes.index(n) for n in prenodes if not n.isroot] ]


        # Filling in the node list. It contains all of the information needed
        # to calculate the likelihoods at each node
        for k,ch in enumerate(postChars):
            [ n for i,n in enumerate(preallocated_arrays["nodelist"]) if leafind[i] ][k][ch] = 1.0
            for i,n in enumerate(preallocated_arrays["nodelist"][:-1]):
                n[nchar] = postnodes.index(postnodes[i].parent)

        # Setting initial node likelihoods to one for calculations
        preallocated_arrays["nodelist"][[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0


    # Calculating the likelihoods for each node in post-order sequence
    for intnode in sorted(set(preallocated_arrays["nodelist"][:-1,nchar])):

        nextli = preallocated_arrays["nodelist"][int(intnode)]

        for ind in np.where(preallocated_arrays["nodelist"][:,nchar]==intnode)[0]:
            li = preallocated_arrays["nodelist"][int(ind)]
            for ch in preallocated_arrays["charlist"]:
                nextli[ch] *= sum([ p[ind][ch,st] for st in preallocated_arrays["charlist"] ] * li[:-1])

    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"


        rootPriors = {state:li for state,li in enumerate(pi)}

        li = sum([ i*rootPriors[n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ])
        logli = math.log(li)

    elif pi == "Equal":
        rootPriors = {state:1.0/nchar for state in range(nchar)}
        li = sum([ float(i)/nchar for i in preallocated_arrays["nodelist"][-1] ])

        logli = math.log(li)

    elif pi == "Fitzjohn":
        rootPriors = { charstate:preallocated_arrays["nodelist"][-1,:-1][charstate]/
                                 sum(preallocated_arrays["nodelist"][-1,:-1]) for
                                 charstate in set(chars) }

        li = sum([ preallocated_arrays["nodelist"][-1,:-1][charstate] *
                     rootPriors[charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        rootPriors = {state:li for state,li in enumerate(qsd(Q))}

        li = sum([ i*rootPriors[n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1])() ])
        logli = math.log(li)
    if returnPi:
        return (logli, rootPriors)
    else:
        return logli




def mk(tree, chars, Q, p = None, pi="Equal", returnPi=False):
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
                       Either a string containing the method or an array
                       of weights. Weights should be given in order.

                       Accepted methods of weighting root:

                       Equal: flat prior
                       Equilibrium: Prior equal to stationary distribution
                         of Q matrix
                       Fitzjohn: Root states weighted by how well they
                         explain the data at the tips.
        returnPi (bool): Whether or not to return the pi used
    """
    chartree = tree.copy()
    chartree.char = None; chartree.likelihoodNode={}
    t = [node.length for node in chartree.descendants()]
    t = np.array(t, dtype=np.double)
    nchar = Q.shape[0]

    # Generating probability matrix for each branch
    if p is None:
        p = np.empty([len(t), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    cyexpokit.dexpm_tree_preallocated_p(Q, t, p) # This changes p in place


    for i, nd in enumerate(chartree.descendants()):
        nd.pmat = p[i] # Assigning probability matrices for each branch
        nd.likelihoodNode = {}
        nd.char = None

    for i, lf in enumerate(chartree.leaves()):
        lf.char = chars[i] # Assigning character states to tips


    for node in chartree.postiter():
        if node.char is not None: # For tip nodes, likelihoods are 1 for observed state and 0 for the rest
            for state in range(nchar):
                node.likelihoodNode[state]=0.0
            node.likelihoodNode[node.char]=1.0
        else:
            for state in range(nchar):
                likelihoodStateN = []
                for ch in node.children:
                    likelihoodStateNCh = []
                    for chState in range(nchar):
                        likelihoodStateNCh.append(ch.pmat[state, chState] * ch.likelihoodNode[chState]) #Likelihood for a certain state = p(stateBegin, stateEnd * likelihood(stateEnd))
                    likelihoodStateN.append(sum(likelihoodStateNCh))
                node.likelihoodNode[state]=np.product(likelihoodStateN)

    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"


        rootPriors = {state:li for state,li in enumerate(pi)}

        li = sum([ i*rootPriors[n] for n,i in chartree.likelihoodNode.iteritems() ])
        logli = math.log(li)

    elif pi == "Equal":
        rootPriors = {state:1.0/nchar for state in range(nchar)}
        li = sum([ i/nchar for i in chartree.likelihoodNode.values() ])

        logli = math.log(li)

    elif pi == "Fitzjohn":
        rootPriors = { charstate:chartree.likelihoodNode[charstate]/
                                 sum(chartree.likelihoodNode.values()) for
                                 charstate in set(chars) }

        li = sum([ chartree.likelihoodNode[charstate] *
                     rootPriors[charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        rootPriors = {state:li for state,li in enumerate(qsd(Q))}

        li = sum([ i*rootPriors[n] for n,i in chartree.likelihoodNode.iteritems() ])
        logli = math.log(li)
    if returnPi:
        return (logli, rootPriors)
    else:
        return logli

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
    nt =  len(tree.descendants())
    p = np.empty([nt, Q.shape[0], Q.shape[1]], dtype = np.double, order="C")

    nchar = Q.shape[0]
    charlist = range(Q.shape[0])

    nodelist,t = _create_nodelist(tree, chars)
    nodelistOrig = nodelist.copy()

    nodelistOrig = nodelist.copy()
    treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
                   tree.leaves()[0].length])
    upperbound = len(tree.leaves())/treelen

    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound}

    def likelihood_function(Qparam):

        # Upper bound on sum of rates is number of tips/tree length
        if sum(Qparam) > var["upperbound"]:
            return 1e255

        nchar = var["charlist"][-1]+1
        param = Qparam[0]

        np.copyto(var["nodelist"], var["nodelistOrig"])

        # Fill in Q matrix
        var["Q"].fill(param)
        var["Q"][np.diag_indices(nchar)] = -param * (nchar-1)
        try:
            return -1 * cy_mk(tree, chars, var["Q"], var["p"], pi, preallocated_arrays=var) # Minimizing negative log-likelihood
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

    charlist = range(Q.shape[0])

    nodelist,t = _create_nodelist(tree, chars)
    nodelistOrig = nodelist.copy()

    nodelistOrig = nodelist.copy()
    treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
                   tree.leaves()[0].length])
    upperbound = len(tree.leaves())/treelen

    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound}


    def likelihood_function(Qparams):
        """
        Number of params equal to binom(nchar, 2). Forward and backwards
        transitions are the same
        """
        # Upper bound on sum of rates is number of tips/tree length
        if sum(Qparams) > var["upperbound"]:
            return 1e255

        var["Q"].fill(0.0) # Re-filling with zeroes

        nchar = len(set(chars))
        params = Qparams[:]
        # Generating symmetric Q matrix

        xs,ys = np.triu_indices(nchar,k=1)
        var["Q"][xs,ys] = params
        var["Q"][ys,xs] = params
        var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)

        np.copyto(var["nodelist"], var["nodelistOrig"])

        try:
            return -1 * cy_mk(tree, chars, var["Q"], p=var["p"],pi=pi,preallocated_arrays=var) # # Minimizing negative log-likelihood
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

    charlist = range(Q.shape[0])

    nodelist,t = _create_nodelist(tree, chars)
    nodelistOrig = nodelist.copy()

    nodelistOrig = nodelist.copy()
    treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
                   tree.leaves()[0].length])
    upperbound = len(tree.leaves())/treelen

    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound}

    def likelihood_function(Qparams):
        """
        Number of params equal to binom(nchar, 2). Forward and backwards
        transitions are the same
        """
        # Upper bound on sum of rates is number of tips/tree length
        if sum(Qparams) > var["upperbound"]:
            return 1e255
        var["Q"].fill(0.0) # Re-filling with zeroes

        nchar = len(set(chars))
        params = Qparams[:]
        # Filling in values for ARD matrix

        var["Q"][np.triu_indices(nchar, k=1)] = params[:len(params)/2]
        var["Q"][np.tril_indices(nchar, k=-1)] = params[len(params)/2:]
        var["Q"][np.diag_indices(nchar)] = 0-np.sum(var["Q"], 1)

        np.copyto(var["nodelist"], var["nodelistOrig"])

        try:
            return -1 * cy_mk(tree, chars, var["Q"], p=var["p"], pi = pi, preallocated_arrays=var) # Minimizing negative log-likelihood
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
                      bounds = [(1e-14,None)])

    q = np.empty([nchar,nchar], dtype=np.double)
    q.fill(optim.x[0])

    q[np.diag_indices(nchar)] = 0 - (q.sum(1)-q[0,0])

    piRates = cy_mk(tree, chars, q, pi=pi, returnPi=True)[1]

    return (q, -1*float(optim.fun), piRates)

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
                      bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))


    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x
    q = q + q.T
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates = cy_mk(tree, chars, q, pi=pi, returnPi=True)[1]

    return (q, -1*float(optim.fun), piRates)


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
                      bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))

    q = np.zeros([nchar,nchar], dtype=np.double)

    q[np.triu_indices(nchar, k=1)] = optim.x[:len(optim.x)/2]
    q[np.tril_indices(nchar, k=-1)] = optim.x[len(optim.x)/2:]
    q[np.diag_indices(nchar)] = 0-np.sum(q, 1)

    piRates = cy_mk(tree, chars, q, pi=pi, returnPi=True)[1]

    return (q, -1*float(optim.fun), piRates)


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
            q,l,piRates = fitMkER(tree, chars, pi="Equal")
            if pi == "Equal":
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}

            # For likelihood calculation of non-flat pi,
            # Calculate Q under flat pi, then use the pi from that Q
            # to determine likelihood
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l,piRates = fitMkER(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
            elif pi =="Fitzjohn":
                pivals = cy_mk(tree, chars, q, pi="Fitzjohn", returnPi=True)[1]
                q,l,piRates = fitMkER(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}

        elif Q == "Sym":
            q,l,piRates = fitMkSym(tree, chars, pi="Equal")
            if pi == "Equal":
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l,piRates = fitMkSym(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
            elif pi =="Fitzjohn":
                pivals = cy_mk(tree, chars, q, pi="Fitzjohn", returnPi=True)[1]
                q,l,piRates = fitMkSym(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
        elif Q == "ARD":
            q,l,piRates = fitMkARD(tree, chars, pi="Equal")
            if pi == "Equal":
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
            elif pi == "Equilibrium":
                pivals = qsd(q)
                q,l,piRates = fitMkARD(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
            elif pi =="Fitzjohn":
                pivals = cy_mk(tree, chars, q, pi="Fitzjohn", returnPi=True)[1]

                pivals = np.array([ pivals[ch] for ch in set(chars)], dtype=np.double)

                q,l,piRates = fitMkARD(tree, chars, pi=pivals)
                return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}
        else:
            raise ValueError("Q str must be one of: 'Equal', 'Sym', 'ARD'")

    else:
        assert str(type(Q)) == "<type 'numpy.ndarray'>", "Q must be str or numpy array"
        assert len(Q[0]) == len(set(chars)), "Supplied Q has wrong dimensions"

        l,piRates = cy_mk(tree, chars, Q, pi=pi, returnPi=True)
        q = Q

        return {key:val for key, val in zip(["Q", "Log-likelihood","pi"], [q,l,piRates])}

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
