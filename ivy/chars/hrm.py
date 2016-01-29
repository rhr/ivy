# Hidden-rates model
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
np.seterr(invalid="warn")

def hrm_mk(tree, chars, Q, nregime, p=None, pi="Fitzjohn",returnPi=False,
          preallocated_arrays=None):
    """
    Note: this version calculates likelihoods at each node.
    Other version calculates probabilities at each node to match
    corHMM
    Return log-likelihood of hidden-rates-model mk as described in
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
        preallocated_arrays (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays
    """
    nchar = Q.shape[0]
    nobschar = nchar/nregime
    if preallocated_arrays is None:
        # Creating arrays to be used later
        preallocated_arrays = {}
        preallocated_arrays["charlist"] = range(Q.shape[0])
        preallocated_arrays["t"] = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)

    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place

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

        # Q matrix is in the form of "0S, 1S, 0F, 1F" etc. Probabilities
        # set to 1 for all hidden states of the observed state.
        for k,ch in enumerate(postChars):
            # Indices of hidden rates of observed state. These will all be set to 1
            hiddenChs = [y + ch for y in [x * nobschar for x in range(nregime) ]]
            [ n for i,n in enumerate(preallocated_arrays["nodelist"]) if leafind[i] ][k][hiddenChs] = 1.0
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
                   charstate in range(nchar) ])

        li = sum([ preallocated_arrays["nodelist"][-1,:-1][charstate] *
                     preallocated_arrays["root_priors"][charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(preallocated_arrays["root_priors"],qsd(Q))
        li = sum([ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ])
        logli = math.log(li)
    if returnPi:
        return (logli, {k:v for k,v in enumerate(preallocated_arrays["root_priors"])})
    else:
        return logli



def create_likelihood_function_hrm_mk(tree, chars, nregime, Qtype, pi="Fitzjohn",
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
        Qtype (str): ARD only
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

    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))

    # Empty Q matrix
    Q = np.zeros([nchar,nchar], dtype=np.double)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    # Empty likelihood array
    nodelist,t,childlist = _create_hrmnodelist(tree, chars, nregime)
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
        else:
            raise ValueError, "Qtype must be ARD"
        for char in range(nobschar):
            hiddenchar =  [y + char for y in [x * nobschar for x in range(nregime) ]]
            for char2 in [ ch for ch in range(nobschar) if not ch == char ]:
                hiddenchar2 =  [y + char2 for y in [x * nobschar for x in range(nregime) ]]
                rs = [Q[ch1, ch2] for ch1, ch2 in zip(hiddenchar, hiddenchar2)]
                if not(rs == sorted(rs)):
                    return var["nullval"]
        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if min:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_mk(tree, chars, var["Q"],nregime,  p=var["p"], pi = pi, preallocated_arrays=var)
            if not np.isnan(logli):
                return x * logli# Minimizing negative log-likelihood
            else:
                return var["nullval"]
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function




def fit_hrm_mkARD(tree, chars, nregime, pi="Fitzjohn"):
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
    x0 = [.10] * n_Q_params(nobschar, nregime)

    mk_func = create_likelihood_function_hrm_mk(tree, chars, nregime=nregime,
                                                 Qtype="ARD", pi=pi)
    optim = minimize(mk_func, x0, method="SLSQP",
                      bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))

    q = np.zeros([nchar,nchar], dtype=np.double)
    i = 0
    for rC in range(nregime):
        for charC in range(nobschar):
          for rR in range(nregime):
                for charR in range(nobschar):
                    if not ((rR == rC) and (charR == charC)):
                        if ((rR == rC) or ((charR == charC)) and (rR+1 == rC or rR-1 == rC)):
                            q[charR+rR*nobschar, charC+rC*nobschar] = optim.x[i]
                            i += 1
    q[np.diag_indices(nchar)] = -np.sum(q, 1)

    piRates = hrm_mk(tree, chars, q, nregime, pi=pi, returnPi=True)[1]

    return (q, -1*float(optim.fun), piRates)

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

def fill_Q_matrix(nobschar, nregime, wrparams, brparams, Qtype="ARD"):
    """
    Fill a Q matrix with nchar*nregime rows and cols with values from Qparams

    Args:
        nchar (int): number of observed characters
        nregime (int): number of hidden rates per character
        wrparams (list): List of unique Q values for within-regime transitions,
          in order as they appear in columnwise iteration
        brparams (list): list of unique Q values for between-regime transition,
          in order as they appear in columnwise iteration
    Returns:
        array: Q-matrix with values filled in. Check to make sure values
          have been filled in properly
    """
    Q = np.zeros([nobschar*nregime, nobschar*nregime])
    assert Qtype in ["ARD", "Simple"]
    grid = np.zeros([(nobschar*nregime)**2, 4], dtype=int)
    grid[:,0] = np.tile(np.repeat(range(nregime), nobschar), nobschar*nregime)
    grid[:,1] = np.repeat(range(nregime), nregime*nobschar**2)
    grid[:,2] = np.tile(range(nobschar), nregime**2*nobschar)
    grid[:,3] = np.tile(np.repeat(range(nobschar), nregime*nobschar), nregime)
    if Qtype == "ARD":
        wrcount = 0
        brcount = 0
        for i, qcell in enumerate(np.nditer(Q, order="C", op_flags=["readwrite"])):
            cell = grid[i]
            if (cell[0] == cell[1]) and cell[2] != cell[3]:
                qcell[...] = wrparams[wrcount]
                wrcount += 1
            elif(cell[0] in [cell[1]+1, cell[1]-1] and cell[2] == cell[3] ):
                qcell[...] = brparams[brcount]
                brcount += 1
        Q[np.diag_indices(nobschar*nregime)] = np.sum(Q, axis=1)*-1
    elif Qtype == "Simple":
        for i,qcell in enumerate(np.nditer(Q, order="C", op_flags=["readwrite"])):
            cell = grid[i]
            if (cell[0] == cell[1]) and cell[2] != cell[3]:
                qcell[...] = wrparams[cell[0]]
            elif(cell[0] in [cell[1]+1, cell[1]-1] and cell[2] == cell[3] ):
                qcell[...] = brparams[0]
        Q[np.diag_indices(nobschar*nregime)] = np.sum(Q, axis=1) * -1
    return Q


def n_Q_params(nchar, nregime):
    """
    Number of free Q params for a matrix with nchar and nregimes
    """
    Cs = [ (i,j) for i in range(nregime) for j in range(nchar)]
    n = [(i,j) for i in Cs for j in Cs if (i!=j and (i[0] + 1 == j[0] or i[0] - 1 == j[0] or i[0]==j[0]) and (i[0]==j[0] or i[1] == j[1])) ]
    return(len(n))

def _create_hrmnodelist(tree, chars, nregime):
    """
    Create nodelist. For use in mk function
    """
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nchar = len(set(chars)) * nregime
    nobschar = len(set(chars))

    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    nodelist = np.zeros((nnode, nchar+1))
    childlist = np.zeros(nnode, dtype=object)
    leafind = [ n.isleaf for n in tree.postiter()]

    for k,ch in enumerate(postChars):
        hiddenChs = [y + ch for y in [x * nobschar for x in range(nregime) ]]
        [ n for i,n in enumerate(nodelist) if leafind[i] ][k][hiddenChs] = 1.0
    for i,n in enumerate(nodelist[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)
        childlist[i] = [ nod.pi for nod in postnodes[i].children ]
    childlist[i+1] = [ nod.pi for nod in postnodes[i+1].children ] # Add the root to the childlist array

    # Setting initial node likelihoods to one for calculations
    nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

    return nodelist,t,childlist


def hrm_back_mk(tree, chars, Q, nregime, p=None, pi="Fitzjohn",returnPi=False,
                preallocated_arrays=None, tip_states=None, returnnodes=False):
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
        preallocated_arrays (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays
    """
    nchar = Q.shape[0]
    nobschar = nchar/nregime
    if preallocated_arrays is None:
        # Creating arrays to be used later
        preallocated_arrays = {}
        t = [node.length for node in tree.postiter() if not node.isroot]
        t = np.array(t, dtype=np.double)
        preallocated_arrays["charlist"] = range(Q.shape[0])
        preallocated_arrays["t"] = t

    if p is None: # Instantiating empty array
        p = np.empty([len(preallocated_arrays["t"]), Q.shape[0], Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, preallocated_arrays["t"], p) # This changes p in place

    if len(preallocated_arrays.keys())==2:
        # Creating more arrays
        nnode = len(tree.descendants())+1
        preallocated_arrays["nodelist"] = np.zeros((nnode, nchar+1))
        preallocated_arrays["childlist"] = np.zeros(nnode, dtype=object)
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
            [ n for i,n in enumerate(preallocated_arrays["nodelist"]) if leafind[i] ][k][hiddenChs] = 1.0/nregime
        for i,n in enumerate(preallocated_arrays["nodelist"][:-1]):
            n[nchar] = postnodes.index(postnodes[i].parent)
            preallocated_arrays["childlist"][i] = [ nod.pi for nod in postnodes[i].children ]
        preallocated_arrays["childlist"][i+1] = [ nod.pi for nod in postnodes[i+1].children ]

        # Setting initial node likelihoods to 1.0 for calculations
        preallocated_arrays["nodelist"][[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

        # Empty array to store root priors
        preallocated_arrays["root_priors"] = np.empty([nchar], dtype=np.double)
        preallocated_arrays["nodelist-up"] = preallocated_arrays["nodelist"].copy()
        preallocated_arrays["t_Q"] = Q
        preallocated_arrays["p_up"] = p.copy()
        preallocated_arrays["v"] = np.zeros([nchar])
        preallocated_arrays["tmp"] = np.zeros([nchar+1])
        preallocated_arrays["motherRow"] = np.zeros([nchar+1])

    leafind = [ n.isleaf for n in tree.postiter()]
    if tip_states is not None:
        leaf_rownums = [i for i,n in enumerate(leafind) if n]
        tip_states = preallocated_arrays["nodelist"][leaf_rownums][:,:-1] * tip_states[:,:-1]
        tip_states = tip_states/np.sum(tip_states,1)[:,None]

        preallocated_arrays["nodelist"][leaf_rownums,:-1] = tip_states

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
                   charstate in range(nchar) ])

        li = sum([ preallocated_arrays["nodelist"][-1,:-1][charstate] *
                     preallocated_arrays["root_priors"][charstate] for charstate in set(chars) ])
        logli = math.log(li)
    elif pi == "Equilibrium":
        # Equilibrium pi from the stationary distribution of Q
        np.copyto(preallocated_arrays["root_priors"],qsd(Q))
        li = sum([ i*preallocated_arrays["root_priors"][n] for n,i in enumerate(preallocated_arrays["nodelist"][-1,:-1]) ])
        logli = math.log(li)

    # Transposal of Q for up-pass now that down-pass is completed
    np.copyto(preallocated_arrays["t_Q"], Q)
    preallocated_arrays["t_Q"] = np.transpose(preallocated_arrays["t_Q"])
    preallocated_arrays["t_Q"][np.diag_indices(nchar)] = 0
    preallocated_arrays["t_Q"][np.diag_indices(nchar)] = -np.sum(preallocated_arrays["t_Q"], 1)
    preallocated_arrays["t_Q"] = np.ascontiguousarray(preallocated_arrays["t_Q"])
    cyexpokit.dexpm_tree_preallocated_p(preallocated_arrays["t_Q"], preallocated_arrays["t"], preallocated_arrays["p_up"])
    preallocated_arrays["nodelist-up"][:,:-1] = 1.0
    preallocated_arrays["nodelist-up"][-1] = preallocated_arrays["nodelist"][-1]

    ni = len(preallocated_arrays["nodelist-up"]) - 2

    root_marginal =  ivy.chars.mk.qsd(Q) # Change to Fitzjohn Q?

    for n in preallocated_arrays["nodelist-up"][::-1][1:]:
        curRow = n[:-1]
        motherRowNum = int(n[nchar])
        np.copyto(preallocated_arrays["motherRow"], preallocated_arrays["nodelist-up"][int(motherRowNum)])
        sisterRows = [ (preallocated_arrays["nodelist-up"][i],i) for i in preallocated_arrays["childlist"][motherRowNum] if not i==ni]

        # If the mother is the root...
        if preallocated_arrays["motherRow"][nchar] == 0.0:
            # The marginal of the root
            np.copyto(preallocated_arrays["v"],root_marginal) # Only need to calculate once
        else:
            # If the mother is not the root, calculate prob. of being in any state
            # Use transposed matrix
            np.dot(preallocated_arrays["p_up"][motherRowNum], preallocated_arrays["nodelist-up"][motherRowNum][:nchar], out=preallocated_arrays["v"])
        for s in sisterRows:
            # Use non-transposed matrix
            np.copyto(preallocated_arrays["tmp"], preallocated_arrays["nodelist"][s[1]])
            preallocated_arrays["tmp"][:nchar] = preallocated_arrays["tmp"][:-1]/sum(preallocated_arrays["tmp"][:nchar])
            preallocated_arrays["v"] *= np.dot(p[s[1]], preallocated_arrays["tmp"][:nchar])
        preallocated_arrays["nodelist-up"][ni][:nchar] = preallocated_arrays["v"]
        ni -= 1
    out = [preallocated_arrays["nodelist-up"][[ t.pi for t in tree.leaves() ]], logli]
    if returnnodes:
        out.append(preallocated_arrays["nodelist-up"])
    return out

def hrm_multipass(tree, chars, Q, nregime, pi="Fitzjohn", preallocated_arrays=None,
                  p = None, returntips=False, returnnodes=False):
    """
    For given tree, chars, and Q, perform up and downpasses, re-assigning
    likelihoods at tips until no further improvement of likelihood is made

    Max. 50 iterations
    """

    t, l = hrm_back_mk(tree, chars, Q, nregime, pi=pi)
    t = np.array(t)
    l = np.array(l)

    for i in range(50):
        t_n, l_n = hrm_back_mk(tree, chars, Q, nregime, pi=pi, tip_states=t, preallocated_arrays=preallocated_arrays,
        p = p)
        if np.isclose(l_n, l):
            break
        else:
            np.copyto(t,t_n)
            np.copyto(l,l_n)
        if preallocated_arrays is not None:
            np.copyto(preallocated_arrays["nodelist"], preallocated_arrays["nodelistOrig"])
            preallocated_arrays["root_priors"].fill(1.0)
    out = [l]
    if returntips:
        out.append(t_n)
    if returnnodes:
        out.append(hrm_back_mk(tree, chars, Q, nregime, pi=pi, tip_states=t, preallocated_arrays=preallocated_arrays,
        p = p, returnnodes=True)[2])
    return out

def create_likelihood_function_hrmmultipass_mk(tree, chars, nregime, Qtype,
                                      pi = "Fitzjohn", min = True):
    """
    Create a function that takes values for Q and returns likelihood after
    performing multiple passes

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
    if min:
        nullval = np.inf
    else:
        nullval = -np.inf
    nchar = len(set(chars)) * nregime
    nt =  len(tree.descendants())
    charlist = range(nchar)
    nobschar = len(set(chars))
    # Empty Q matrix
    Q = np.zeros([nchar,nchar], dtype=np.double)
    t_Q = Q.copy()
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    # Empty likelihood array
    nodelist,t,childlist = _create_hrmnodelist(tree, chars, nregime)
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
           "root_priors":rootpriors, "nullval":nullval, "t_Q":t_Q,
           "p_up":p.copy(), "v":np.zeros([nchar]), "tmp":np.zeros([nchar+1]),
           "motherRow":np.zeros([nchar+1]), "childlist":childlist}
    var["nodelist-up"] =var["nodelist"].copy()
    def likelihood_function(Qparams):
        # Enforcing upper bound on parameters
        if (sum(Qparams) > (var["upperbound"]*2)) or any(Qparams < 0):
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
        else:
            raise ValueError, "Qtype must be ARD"
        for char in range(nobschar):
            hiddenchar =  [y + char for y in [x * nobschar for x in range(nregime) ]]
            for char2 in [ ch for ch in range(nobschar) if not ch == char ]:
                hiddenchar2 =  [y + char2 for y in [x * nobschar for x in range(nregime) ]]
                rs = [Q[ch1, ch2] for ch1, ch2 in zip(hiddenchar, hiddenchar2)]
                if not(rs == sorted(rs)):
                    return var["nullval"]


        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if min:
            x = -1
        else:
            x = 1
        try:
            logli =  hrm_multipass(tree, chars, var["Q"], nregime, p=var["p"], pi = pi, preallocated_arrays=var)
            if not np.isnan(logli):
                return x * logli[0] # Minimizing negative log-likelihood
            else:
                return var["nullval"]
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function
