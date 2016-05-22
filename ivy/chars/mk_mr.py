# Mk multi regime models
import math
import random
import itertools

import numpy as np
import scipy
import pymc
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom

from ivy.chars.expokit import cyexpokit



def mk_multi_regime(tree, chars, Qs, locs, pi="Equal", returnPi=False,
                     ar = None):
    """
    Calculate likelhiood of mk model with BAMM-like multiple regimes

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
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
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = Qs[0].shape[0]
    if ar is None:
        # Creating arrays to be used later
        ar = create_mkmr_ar(tree, chars,Qs.shape[2],findmin=True)

    inds = [0]*len(ar["t"])

    for l, a in enumerate(locs):
        for n in a:
            inds[n-1] = l

    # Creating probability matrices from Q matrices and branch lengths
    # inds indicates which Q matrix to use for which branch
    cyexpokit.dexpm_treeMulti_preallocated_p_log(Qs,ar["t"], ar["p"], np.array(inds)) # This changes p in place


    # Calculating the likelihoods for each node in post-order sequence
    cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"])
    # The last row of nodelist contains the likelihood values at the root

    # Applying the correct root prior
    if type(pi) != str:
        assert len(pi) == nchar, "length of given pi does not match Q dimensions"
        assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
        assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"

        np.copyto(ar["root_priors"], pi)

        rootliks = ([ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ])

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
        return (logli, {k:v for k,v in enumerate(ar["root_priors"])})
    else:
        return logli


def create_mkmr_ar(tree, chars,nregime,findmin = True):
    """
    Create preallocated arrays. For use in mk function

    Nodelist = edgelist of nodes in postorder sequence
    """
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
    Q = np.zeros([nregime,nchar, nchar], dtype=np.double)
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
    tmp_ar = np.zeros(nchar) # Used for storing calculations

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar}
    return var

def create_likelihood_function_multimk(tree, chars, Qtype, nregime, pi="Equal",
                                  findmin = True):
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf

    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = range(nchar)

    # Number of parameters per Q matrix
    n_qp = nchar**2-nchar

    # Empty Q matrix
    Q = np.zeros([nregime,nchar,nchar], dtype=np.double)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    # Empty likelihood array
    var = create_mkmr_ar(tree, chars,nregime,findmin)
    def likelihood_function(Qparams, locs):
        # Enforcing upper bound on parameters
        Qparams = [float(qp) for qp in Qparams]
        # # TODO: replace with sum of each Q
        if (np.sum(Qparams)/len(locs) > var["upperbound"]) or any([x<=0 for x in Qparams]):
            return var["nullval"]
        # if not (Qparams == sorted(Qparams)):
        #     return var["nullval"]
        # Filling Q matrices:
        Qparams = [Qparams[i:i+n_qp] for i in range(nregime)]

        if Qtype == "ER":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(float(Qparams[i]))
                qmat[np.diag_indices(nchar)] = -Qparams[i] * (nchar-1)
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
            return x * mk_multi_regime(tree, chars, var["Q"], locs, pi = pi, ar=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function


def create_likelihood_function_multimk_mods(tree, chars, mods, pi="Equal",
                                  findmin = True):
    """
    Create a likelihood function for testing the parameter values of user-
    specified models
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf

    nregime = len(mods)

    nchar = len(set(chars))
    nt =  len(tree.descendants())
    charlist = range(nchar)

    nparam = len(set([i for s in mods for i in s]))

    # Empty likelihood array
    var = create_mkmr_ar(tree, chars,nregime,findmin)
    def likelihood_function(Qparams, locs):
        # Enforcing upper bound on parameters
        Qparams = np.insert(Qparams, 0, 1e-15)
        # # TODO: replace with sum of each Q

        if (np.sum(Qparams)/len(locs) > var["upperbound"]) or any([x<=0 for x in Qparams]):
            return var["nullval"]


        # Clearing Q matrices
        var["Q"].fill(0.0)
        # Filling Q matrices:
        fill_model_mr_Q(Qparams, mods, var["Q"])

        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if findmin:
            x = -1
        else:
            x = 1

        try:
            return x * mk_multi_regime(tree, chars, var["Q"], locs, pi = pi, ar=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def fill_model_mr_Q(Qparams, mods, Q):
    """
    Fill a Q-matrix with Qparams corresponding to the model.

    Individual Q matrices are filled rowwise

    Function alters Q in place
    """
    # Filling Q matrix row-wise (skipping diagonals)
    nregime = len(mods)
    nchar=Q.shape[1]
    for regime in range(nregime):
        modcount = 0 # Which mod index are we on
        for r,c in itertools.product(range(nchar),repeat=2):
            if r==c: # Skip diagonals
                pass
            else:
                Q[regime,r,c] = Qparams[mods[regime][modcount]]
                modcount+=1
        np.fill_diagonal(Q[regime], -np.sum(Q[regime], axis=1)) # Filling diagonals


def locs_from_switchpoint(tree, switches, locs=None):
    """
    Given a tree and a single node to be the switchpoint, return an
    array of all node indices in one regime vs the other

    Args:
        tree (Node): Root node of tree
        switches (list): List of internal nodes to act as switchpoints.
          Switchpoint nodes are part of their own regimes.
        locs (np.array): Optional pre-allocated array to store output.
    Returns:
        np.array: Array of indices of all nodes associated with each switchpoint, plus
          all nodes "outside" the switches in the last list
    """
    # Include the root
    switches = switches + [tree]
    # Sort switches by clade size
    switches_c = switches[:]
    switches_c.sort(key=len)

    if locs is None:
        locs = np.zeros(len(switches_c), dtype=object)
    else:
        locs.fill(0) # Clear locs array
    for i,s in enumerate(switches_c):
        # Add descendants of node to location array if they are not
        # already recorded. This approach works because the clade sizes
        # were sorted beforehand
        locs[i] = [n.ni for n in tree[s] if not n.ni in [x for l in locs[:i] for x in l]]
    # Remove the root
    locs[-1] = locs[-1][1:]

    # Return locs in proper order (locations descended from each switch in order,
    # with locations descended from the root last)
    return locs[[switches_c.index(i) for i in switches]]



class SwitchpointMetropolis(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new switchpoint
    """
    def __init__(self, stochastic, tree):
        pymc.Metropolis.__init__(self, stochastic, scale=1., proposal_distribution="prior")
        self.tree = tree
    def propose(self):
        # Jumps can happen to children, parent, or siblings.
        cur_node = self.tree[int(self.stochastic.value)]
        adjacent_nodes = cur_node.children+[cur_node.parent]+cur_node.parent.children
        valid_nodes = [n for n in adjacent_nodes if not (n.isleaf or n.isroot)]
        new = random.choice(valid_nodes)
        self.stochastic.value = new.ni

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value

def make_switchpoint_stoch(tree, name="switch"):
    startingval = random.choice(tree.internals()[1:]).ni
    @pymc.stochastic(dtype=int, name=name)
    def switchpoint_stoch(value = startingval):
        # Flat prior on switchpoint location
        return 0
    return switchpoint_stoch


def mk_multi_bayes(tree, chars, mods=None, pi="Equal", nregime=None):
    """
    Create a Bayesian multi-mk model. User specifies which regime models
    to use and the Bayesian model finds the switchpoints.
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    # Preparations

    if nregime is None:
        nregime = len(mods)
    nchar = len(set(chars))
    if mods is not None:
        nparam = len(set([i for s in mods for i in s]))
    else: # Default to ARD model
        nparam = ((nchar**2)-nchar) * nregime

    # This model has 2 components: Q parameters and switchpoints
    # They are combined in a custom likelihood function

    ###########################################################################
    # Switchpoint:
    ###########################################################################
    # Modeling the movement of the regime shift(s) is the tricky part
    # Regime shifts will only be allowed to happen at a node
    switch = [None]*(nregime-1)
    for regime in range(nregime-1):
        switch[regime]= make_switchpoint_stoch(tree, name="switch_{}".format(regime))
    ###########################################################################
    # Qparams:
    ###########################################################################
    # Each Q parameter is an exponential
    Qparams = [None] * nparam
    for i in range(nparam):
         Qparams[i] = pymc.Exponential(name="Qparam"+str(i), beta=1.0)

    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function

    # Pre-allocating arrays
    locsarray = np.empty([nregime], dtype=object)
    if mods is not None:
        l = create_likelihood_function_multimk_mods(tree=tree, chars=chars,
            mods=mods, pi=pi, findmin=False)
    else:
        l = create_likelihood_function_multimk(tree, chars=chars, Qtype="ARD",
                                               nregime=nregime, pi=pi,
                                               findmin=False)

    @pymc.potential
    def multi_mklik(q = Qparams, switch=switch, name="multi_mklik"):

        locs = locs_from_switchpoint(tree,[tree[int(i)] for i in switch],locsarray)
        return l(q, locs=locs)

    mod = pymc.MCMC(locals())

    for s in switch:
        mod.use_step_method(SwitchpointMetropolis, s, tree)

    return mod
