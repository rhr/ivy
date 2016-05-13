# Mk multi regime models
import math
import random

import numpy as np
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom

from ivy.chars.expokit import cyexpokit



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
    cyexpokit.dexpm_treeMulti_preallocated_p_log(Qs, preallocated_arrays["t"], p, np.array(inds)) # This changes p in place

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
    cyexpokit.cy_mk(preallocated_arrays["nodelist"], p, nchar)
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


def create_likelihood_function_multimk(tree, chars, Qtype, nregime, pi="Equal",
                                  findmin = True):
    if findmin:
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
    ar = create_mk_ar(tree, chars)
    nodelist,t = (ar["nodelist"], ar["t"])
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

        if findmin:
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


class SwitchpointMetropolis(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new switchpoint
    """
    def __init__(self, stochastic, tree):
        pymc.Metropolis.__init__(self, stochastic, scale=1.)
        self.tree = tree
    def propose(self):
        cur_node = self.stochastic.value
        adjacent_nodes = cur_node.children+[cur_node.parent]
        valid_nodes = [n for n in adjacent_nodes if not (n.isleaf or n.isroot)]
        new = random.choice(valid_nodes)
        self.stochastic.value = new

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value

def make_switchpoint_stoch(tree, name="switchpoint_stoch"):
    startingval = random.choice(tree.internals()[1:])
    @pymc.stochastic(dtype=ivy.tree.Node, name=name)
    def switchpoint_stoch(value = startingval):
        # Flat prior on switchpoint location
        return 0
    return switchpoint_stoch
def create_mk_ar(tree, chars, findmin = True):
    """
    Create preallocated arrays. For use in mk function

    Nodelist = edgelist of nodes in postorder sequence
    """
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
    charlist = range(nchar)
    tmp_ar = np.zeros(nchar)
    # Giving internal function access to these arrays.
       # Warning: can be tricky
       # Need to make sure old values
       # Aren't accidentally re-used
    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar}
    return var

def mk_multi_bayes(tree, chars, mods=None, pi="Equal"):
    """
    Create a Bayesian multi-mk model. User specifies which regime models
    to use and the Bayesian model finds the switchpoints.
    """
    # Preparations
    # TODO: generalize
    nregime = 2
    nchar = len(set(chars))

    N = int((nchar ** 2 - nchar))
    # This model has 2 components: Q parameters and a switchpoint
    # They are combined in a custom likelihood function

    ###########################################################################
    # Switchpoint:
    ###########################################################################
    # Modeling the movement of the regime shift(s) is the tricky part
    # Regime shifts will only be allowed to happen at a node
    # Regime shift: Uniform categorical distribution
    # TODO: multiple switchpoints
    switch = make_switchpoint_stoch(tree)
    ###########################################################################
    # Qparams:
    ###########################################################################
    # Unscaled Q param: Dirichlet distribution
    # Setting a Dirichlet prior with Jeffrey's hyperprior of 1/2
    theta = [1.0/2.0]*N

    # One set of Q-parameters per regime
    allQparams_init = np.empty(nregime, dtype=object)
    allQparams_init_full = np.empty(nregime, dtype=object)
    allScaling_factors = np.empty(nregime, dtype=object)
    for i in range(nregime):
        if N != 1:
            allQparams_init[i] = pymc.Dirichlet("allQparams_init"+str(i), theta)
            allQparams_init_full[i] = pymc.CompletedDirichlet("allQparams_init_full"+str(i), allQparams_init[i])
        else: # Dirichlet function does not like creating a distribution
              # with only 1 state. Set it to 1 by hand
            allQparams_init_full[i] = [[1.0]]
        # Exponential scaling factor for Qparams
        allScaling_factors[i] = pymc.Exponential(name="allScaling_factors"+str(i), beta=1.0)
        # Scaled Qparams; we would not expect them to necessarily add
        # to 1 as would be the case in a Dirichlet distribution

    # Regimes are grouped by rows. Each row is a regime.
    @pymc.deterministic(plot=False)
    def Qparams(q=allQparams_init_full, s=allScaling_factors):
        Qs = np.empty([nregime,N])
        for n in range(N):
            for i in range(nregime):
                Qs[i][n] = q[i][0][n]*s[i]
        return Qs
    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function

    # Pre-allocating arrays
    qarray = np.zeros([nregime,N])
    locsarray = np.empty([2], dtype=object)
    l = create_likelihood_function_multimk(tree=tree, chars=chars,
        Qtype="ARD", #TODO: generalize
        pi="Equal", findmin=False, nregime=2)

    @pymc.potential
    def multi_mklik(q = Qparams, switch=switch, name="multi_mklik"):

        locs = locs_from_switchpoint(tree,switch,locsarray)

        # l = discrete.create_likelihood_function_multimk(tree=tree, chars=chars,
        #     Qtype=Qtype, locs = locs,
        #     pi="Equal", findmin=False)
        np.copyto(qarray, q)
        return l(qarray, locs=locs)
    return locals()
