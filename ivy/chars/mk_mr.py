# Mk multi regime models
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import random
import itertools
import types
from math import ceil

import line_profiler
import numpy as np
import scipy
import pymc
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom

from ivy.chars.expokit import cyexpokit

try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]

# def mk_mr(tree, chars, Qs, locs, pi="Equal", returnPi=False,
#                      ar = None):
#     """
#     Calculate likelhiood of mk model with BAMM-like multiple regimes
#
#     Args:
#         tree (Node): Root node of a tree. All branch lengths must be
#           greater than 0 (except root)
#         chars (dict): Dict mapping character states to tip labels.
#           Character states should be coded 0,1,2...
#
#           Can also be a list with tip states in preorder sequence
#         Qs (np.array): Array of instantaneous rate matrices
#         locs (np.array): Array of the same length as Qs containing the
#           node indices that correspond to each Q matrix
#         p (np.array): Optional pre-allocated p matrix
#         pi (str or np.array): Option to weight the root node by given values.
#            Either a string containing the method or an array
#            of weights. Weights should be given in order.
#
#            Accepted methods of weighting root:
#
#            Equal: flat prior
#            Equilibrium: Prior equal to stationary distribution
#              of Q matrix
#            Fitzjohn: Root states weighted by how well they
#              explain the data at the tips.
#     """
#     if type(chars) == dict:
#         chars = [chars[l] for l in [n.label for n in tree.leaves()]]
#     nchar = Qs[0].shape[0]
#     if ar is None:
#         # Creating arrays to be used later
#         ar = create_mkmr_ar(tree, chars,Qs.shape[2],findmin=True)
#
#     inds = [0]*len(ar["t"])
#
#     for l, a in enumerate(locs):
#         for n in a:
#             posti = tree[n].pi
#             inds[posti] = l
#
#     # Creating probability matrices from Q matrices and branch lengths
#     # inds indicates which Q matrix to use for which branch
#     cyexpokit.dexpm_treeMulti_preallocated_p_log(Qs,ar["t"], ar["p"], np.array(inds)) # This changes p in place
#
#
#     # Calculating the likelihoods for each node in post-order sequence
#     cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"])
#     # The last row of nodelist contains the likelihood values at the root
#
#     # Applying the correct root prior
#     if not type(pi) in StringTypes:
#         assert len(pi) == nchar, "length of given pi does not match Q dimensions"
#         assert str(type(pi)) == "<type 'numpy.ndarray'>", "pi must be str or numpy array"
#         assert np.isclose(sum(pi), 1), "values of given pi must sum to 1"
#
#         np.copyto(ar["root_priors"], pi)
#
#         rootliks = ([ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ])
#
#     elif pi == "Equal":
#         ar["root_priors"].fill(1.0/nchar)
#         rootliks = [ i+np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1])]
#
#     elif pi == "Fitzjohn":
#         np.copyto(ar["root_priors"],
#                   [ar["nodelist"][-1,:-1][charstate]-
#                    scipy.misc.logsumexp(ar["nodelist"][-1,:-1]) for
#                    charstate in set(chars) ])
#         rootliks = [ ar["nodelist"][-1,:-1][charstate] +
#                      ar["root_priors"][charstate] for charstate in set(chars) ]
#     elif pi == "Equilibrium":
#         # Equilibrium pi from the stationary distribution of Q
#         np.copyto(ar["root_priors"],qsd(Q))
#         rootliks = [ i + np.log(ar["root_priors"][n]) for n,i in enumerate(ar["nodelist"][-1,:-1]) ]
#     logli = scipy.misc.logsumexp(rootliks)
#     if returnPi:
#         return (logli, {k:v for k,v in enumerate(ar["root_priors"])})
#     else:
#         return logli

def mk_mr_midbranch(tree, chars, Qs, switchpoint, pi="Equal", returnPi=False,
                     ar = None, debug=False, pmask=True):
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
    if len(switchpoint)>0:
        if type(switchpoint[0]) == pymc.PyMCObjects.Stochastic:
            switchpoint = [i.value for i in switchpoint]
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]

    nchar = len(set(chars))

    if ar is None:
        ar = create_mkmr_mb_ar(tree,chars,nregime=Qs.shape[0],findmin=True)
    chars = ar["chars"]
    switchpoint_nodes = [ar["tree_copy"][switchpoint[i][0].id] for i in range(len(switchpoint))]
    locs = locs_from_switchpoint(ar["tree_copy"], switchpoint_nodes,locs=ar["locs"])

    # Reset t values
    ar["t"] = ar["blens"][:]

    # Adjust t to account for presence of switchpoint
    for s in switchpoint:
        sw = ar["tree_copy"][s[0].id].pi #Switchpoint
        sk = ar["tree_copy"][s[0].id].parent.pi #Switchpoint's parent
        ar["t"][sw] = s[1]
        ar["t"][sk] =  ar["tree_copy"][s[0].id].parent.length - s[1]

    # inds corresponds to the tree in postorder sequence, with each value
    # corresponding to the regime that node is in
    inds = [0]*len(ar["t"])
    for l, a in enumerate(locs):
        for n in a:
            posti = ar["pre_post"][n]
            inds[posti] = l
    (Qs != ar["prev_Q"]).any(axis=(1,2), out=ar["Qdif"])
    indsmask = inds[:]

    #Which Qs are different?

    np.logical_or.reduce((ar["prev_inds"]!=inds, ar["t"]!=ar["prev_t"],ar["Qdif"][indsmask]),out=ar["pmask"]) # Which p-matrices do we need to recalcualte?
    np.copyto(ar["prev_t"],ar["t"])
    np.copyto(ar["prev_inds"], inds)
    np.copyto(ar["prev_Q"], Qs)

    # Creating probability matrices from Q matrices and branch lengths
    # inds indicates which Q matrix to use for which branch
    if not pmask:
        ar["pmask"].fill(True)
    cyexpokit.dexpm_treeMulti_preallocated_p_log(Qs,ar["t"], ar["p"], np.array(inds), pmask=ar["pmask"]) # This changes p in place
    # Calculating the likelihoods for each node in post-order sequence

    np.copyto(ar["nodelist"], ar["nodelistOrig"]) # clearing the nodelist
    cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"])
    # The last row of nodelist contains the likelihood values at the root
    # Applying the correct root prior
    if not type(pi) in StringTypes:
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
    ar["lastQ"] = Qs
    ar["lastswitch"] = switchpoint
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
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double) # t is in POSTORDER SEQUENCE
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
    charlist = list(range(nchar))
    tmp_ar = np.zeros(nchar) # Used for storing calculations

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar}
    return var

def create_mkmr_mb_ar(tree, chars,nregime,findmin = True):
    """
    Preallocated arrays for midbranch mk_mr

    Create preallocated arrays. For use in mk_mr midbranch function

    Nodelist = edgelist of nodes in postorder sequence
    """
    if type(chars) != dict:
        chars = {tree.leaves()[i].label:v for i,v in enumerate(chars)}
    tree_copy = tree.copy()
    for n in tree_copy:
        n.cladesize = len(n)
    # Here we break up the tree so that each node has a "knee"
    # for a parent. This knee starts with a length of 0, effectively
    # making it nonexistant, but can have its length changed
    # to act as a switchpoint.
    for n in tree_copy.descendants():
        n.bisect_branch(1e-15)
    tree_copy.reindex()

    chars = [chars[l] for l in [n.label for n in tree_copy.leaves()]]
    blens = np.array([node.length for node in tree_copy.postiter() if not node.isroot])
    t = np.array(blens, dtype=np.double)
    prev_t = t.copy()
    prev_t.fill(np.inf)
    nt = len(t)
    nchar = len(set(chars))
    preleaves = [ n for n in tree_copy.preiter() if n.isleaf ]
    postleaves = [n for n in tree_copy.postiter() if n.isleaf ]
    postnodes = list(tree_copy.postiter())

    edgelist = [-np.inf] * len(tree_copy)
    for i in range(len(edgelist)-1):
        edgelist[i] = postnodes[i].parent.pi

    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    nodelist = np.zeros((nnode, nchar+1))
    nodelist.fill(-np.inf) # the log of 0 is negative infinity
    leafind = [ n.isleaf for n in tree_copy.postiter()]

    for k,ch in enumerate(postChars):
        [ n for i,n in enumerate(nodelist) if leafind[i] ][k][ch] = np.log(1.0)
    nodelist[:,-1] = edgelist

    # Setting initial node likelihoods to log one for calculations
    nodelist[[ i for i,b in enumerate(leafind) if not b],:-1] = np.log(1.0)

    # Empty Q matrix
    Q = np.zeros([nregime,nchar, nchar], dtype=np.double)
    prev_Q = Q.copy()
    prev_Q.fill(np.inf)
    # Empty p matrix
    p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
    pmask = np.array([True]*len(t))
    nodelistOrig = nodelist.copy()
    rootpriors = np.empty([nchar], dtype=np.double)
    if findmin:
        nullval = np.inf
    else:
        nullval = -np.inf
    treelen = sum([ n.length for n in tree_copy.leaves()[0].rootpath() if n.length]+[
                   tree_copy.leaves()[0].length])
    upperbound = len(tree_copy.leaves())/treelen
    charlist = list(range(nchar))
    tmp_ar = np.zeros(nchar) # Used for storing calculations

    pre_post = {n.ni:n.pi for n in tree_copy} # The postorder index that corresponds to each preorder index
    prev_inds = np.array([None]*len(t)) #Keeping track of locations from previous call
    Qdif = np.ones([nregime], dtype=bool) # Array for keeping track of changes to Q
    locs = np.zeros(nregime, dtype=object)

    max_children = max(len(n.children) for n in tree_copy)

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar,
           "tree_copy":tree_copy,"chars":chars,"pre_post":pre_post,"prev_Q":prev_Q,
           "prev_t":prev_t, "blens":blens,"pmask":pmask,
           "prev_inds":prev_inds, "Qdif":Qdif,"locs":locs}
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
    charlist = list(range(nchar))

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
            raise ValueError("Qtype must be one of: ER, Sym, ARD")
        # Resetting the values in these arrays
        np.copyto(var["nodelist"], var["nodelistOrig"])
        var["root_priors"].fill(1.0)

        if findmin:
            x = -1
        else:
            x = 1

        try:
            return x * mk_mr(tree, chars, var["Q"], locs, pi = pi, ar=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def lf_mk_mr_midbranch(tree, chars, Qtype, nregime, pi="Equal",
                                  findmin = True):

    if type(chars) == dict:
        chardict = chars
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    else:
        chardict = {tree.leaves()[i].label:v for i,v in enumerate(chars)}

    nchar = len(set(chars))


    # Empty likelihood array
    var = create_mkmr_mb_ar(tree, chardict,nregime,findmin)

    # Number of parameters per Q matrix
    n_qp = nchar**2-nchar


    def likelihood_function(Qparams, switchpoint):
        # Enforcing upper bound on parameters
        Qparams = [float(qp) for qp in Qparams]
        # # TODO: replace with sum of each Q
        if (np.sum(Qparams)/nregime > var["upperbound"]) or any([x<=0 for x in Qparams]):
            return var["nullval"]
        # if not (Qparams == sorted(Qparams)):
        #     return var["nullval"]
        # Filling Q matrices:
        Qparams = [Qparams[i*n_qp:i*n_qp+n_qp] for i in range(nregime)]

        if Qtype == "ER":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(float(Qparams[i]))
                qmat[np.diag_indices(nchar)] = -Qparams[i] * (nchar-1)
        elif Qtype == "ARD":
            for i,qmat in enumerate(var["Q"]):
                qmat.fill(0.0) # Re-filling with zeroes
                qmat[np.triu_indices(nchar, k=1)] = Qparams[i][:len(Qparams[i])//2]
                qmat[np.tril_indices(nchar, k=-1)] = Qparams[i][len(Qparams[i])//2:]
                qmat[np.diag_indices(nchar)] = 0-np.sum(qmat, 1)
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
            return x * mk_mr_midbranch(tree, chars, var["Q"], switchpoint, pi = pi, ar=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def create_likelihood_function_multimk_mods(tree, chars, mods, pi="Equal",
                                  findmin = True, orderedparams=True):
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
    charlist = list(range(nchar))

    nparam = len(set([i for s in mods for i in s]))

    # Empty likelihood array
    var = create_mkmr_ar(tree, chars,nregime,findmin)
    def likelihood_function(Qparams, locs):
        # Enforcing upper bound on parameters
        Qparams = np.insert(Qparams, 0, 1e-15)
        # # TODO: replace with sum of each Q

        if (np.sum(Qparams)/len(locs) > var["upperbound"]) or any([x<=0 for x in Qparams]):
            return var["nullval"]

        if orderedparams:
            if any(Qparams != sorted(Qparams)):
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
            return x * mk_mr(tree, chars, var["Q"], locs, pi = pi, ar=var) # Minimizing negative log-likelihood
        except ValueError: # If likelihood returned is 0
            return var["nullval"]

    return likelihood_function

def lf_mk_mr_midbranch_mods(tree, chars, mods, pi="Equal",
                                  findmin = True, orderedparams=True):
    """
    Create a likelihood function for testing the parameter values of user-
    specified models with switchpoints allowed in the middle of branches
    """
    if type(chars) == dict:
        chardict = chars
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    else:
        chardict = {tree.leaves()[i].label:v for i,v in enumerate(chars)}

    nregime = len(mods)

    nparam = len(set([i for s in mods for i in s]))

    # Empty likelihood array
    var = create_mkmr_mb_ar(tree, chardict,nregime,findmin)
    def likelihood_function(Qparams, switchpoint):
        # Enforcing upper bound on parameters
        Qparams = np.insert(Qparams, 0, 1e-15)
        # # TODO: replace with sum of each Q

        if (np.sum(Qparams)/nregime > var["upperbound"]) or any([x<=0 for x in Qparams]):
            return var["nullval"]

        if orderedparams:
            if any(Qparams != sorted(Qparams)):
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
            return x * mk_mr_midbranch(tree, chars, var["Q"], switchpoint, pi = pi, ar=var) # Minimizing negative log-likelihood
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
        for r,c in itertools.product(list(range(nchar)),repeat=2):
            if r==c: # Skip diagonals
                pass
            else:
                Q[regime,r,c] = Qparams[mods[regime][modcount]]
                modcount+=1
        np.fill_diagonal(Q[regime], -np.sum(Q[regime], axis=1)) # Filling diagonals


def locs_from_switchpoint(tree, switches, locs=None):
    """
    Given a tree and a list of nodes to be switchpoints, return an
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
    # We need the "cladesize" attribute to sort the clades. If it does not
    # exist on the tree, create it
    if not hasattr(tree, "cladesize"):
        for n in tree:
            n.cladesize = len(n)
    switches_c = switches[:]
    switches_c.sort(key=lambda x: x.cladesize)
    if locs is None:
        locs = np.zeros(len(switches_c), dtype=object)
    else:
        locs.fill(0) # Clear locs array
    for i,s in enumerate(switches_c):
        t_l = []
        # Add descendants of node to location array if they are not
        # already recorded. This approach works because the clade sizes
        # were sorted beforehand
        viewed = set([x for l in locs[:i] for x in l])
        for n in tree[s]:
            if not n.ni in viewed:
                t_l.append(n.ni)
        locs[i] = t_l
    # Remove the root
    locs[-1] = locs[-1][1:]

    # Return locs in proper order (locations descended from each switch in order,
    # with locations descended from the root last)
    return locs[[switches_c.index(i) for i in switches]]



class ModelOrderMetropolis(pymc.Metropolis):
    """
    Custom step method for model order
    """
    def __init__(self, stochastic):
        pymc.Metropolis.__init__(self, stochastic, scale=1., proposal_distribution="prior")
    def propose(self):
        cur_order = self.stochastic.value
        indexes_to_swap = random.sample(range(len(cur_order)),2)

        new_order = cur_order[:]
        # Swap values
        new_order[indexes_to_swap[0]], new_order[indexes_to_swap[1]] = new_order[indexes_to_swap[1]], new_order[indexes_to_swap[0]]

        self.stochastic.value = new_order
    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value
    def competence(self):
        return 0


class SwitchpointMetropolis(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new switchpoint
    """
    def __init__(self, stochastic, tree, seg_map, stepsize=0.05, seglen=0.02):
        pymc.Metropolis.__init__(self, stochastic, scale=0,proposal_sd=1, proposal_distribution="prior")
        root_to_tip_length = sum([n.length for n in list(tree.leaves()[0].rootpath())[:-1]+[tree.leaves()[0]]])
        self.tree = tree
        self.seg_map = seg_map
        self.stepsize = stepsize * root_to_tip_length
        self.seg_size = seglen * root_to_tip_length
    def propose(self):
        # Following BAMM, switchpoint movements can be either global or
        # local, with global switch happening 1/10 times

        if (random.choice(range(10))==0):
            self.global_step()
        else:
            self.local_step()

    def local_step(self):
        prev_location = self.stochastic.value
        cur_node = prev_location[0]
        cur_len = prev_location[1]
        direction = random.choice([-1,1])
        step_size = np.random.uniform(high=self.stepsize)
        step_size = round_step_size(step_size, self.seg_size)
        while True:
            if direction == 1: # Rootward
                if step_size < (cur_node.length - cur_len): # Stay on same branch
                    new_location = (cur_node, cur_len+step_size+1e-15)
                    break
                else: # If step goes past the parent
                    if not cur_node.parent.isroot: # Jump to parent node
                        step_size = step_size - ((cur_node.length//self.seg_size)*self.seg_size-cur_len)
                        cur_node = cur_node.parent
                        cur_len = 1e-15
                    else: # If parent is root, jump to a child of the root
                        step_size = step_size - ((cur_node.length//self.seg_size)*self.seg_size-cur_len)
                        valid_nodes = [n for n in cur_node.parent.children if n != cur_node]
                        cur_node = random.choice(valid_nodes)

                        cur_len = (cur_node.length//self.seg_size)*self.seg_size
                        direction = -1
            else: # tipward
                # Stay on same branch
                if step_size < cur_len:
                    new_location = (cur_node, cur_len-step_size)
                    break
                else: # Move to new branch
                    if not cur_node.isleaf: #Move to child
                        step_size = step_size - cur_len + 1e-15
                        valid_nodes = cur_node.children
                        cur_node = random.choice(valid_nodes)
                        cur_len = (cur_node.length//self.seg_size)*self.seg_size
                    else: # Bounce up from tip
                        step_size = step_size - cur_len
                        cur_len = 1e-15
                        direction = 1

        self.stochastic.value = new_location

    def global_step(self):
        self.stochastic.value = random_tree_location(self.seg_map)

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value
    def competence(self):
        return 0

def round_step_size(step_size, seg_size):
    """
    Round step_size to the nearest segment
    """
    if (step_size%seg_size) > (seg_size/2):
        return step_size + (seg_size-(step_size%seg_size)) + 1e-15
    else:
        return step_size - (step_size%seg_size) + 1e-15

def tree_map(tree, seglen=0.02):
    """
    Make a map of the tree cut into segments of size (seglen*root_to_tip_length)
    """
    root_to_tip_length = sum([n.length for n in list(tree.leaves()[0].rootpath())[:-1]+[tree.leaves()[0]]])
    seg_size = seglen * root_to_tip_length
    seg_map = []
    seen = [tree]
    for node in tree.descendants():
        cur_len = node.length
        nseg = int(ceil(cur_len/seg_size))
        for n in range(nseg):
            seg_map.append((node, n*seg_size+1e-15))
    return seg_map


def random_tree_location(seg_map):
    """
    Select random location on tree with uniform probability

    Returns:
        tuple: node and float, which represents the how far along the branch to
          the parent node the switch occurs on
    """
    i = random.choice(range(len(seg_map)))
    return seg_map[i]


def make_switchpoint_stoch(seg_map, name=str("switch")):
    startingval = random_tree_location(seg_map)
    @pymc.stochastic(dtype=object, name=name)
    def switchpoint_stoch(value = startingval):
        # Flat prior on switchpoint location
        return 0
    return switchpoint_stoch

def make_modelorder_stoch(mods, name=str("modorder")):
    startingval = mods
    @pymc.stochastic(dtype=tuple, name=name)
    def modelorder_stoch(value=startingval):
        return 0
    return modelorder_stoch

def mk_multi_bayes(tree, chars, mods=None, pi="Equal", nregime=None, db=None,
                   dbname=None,seglen=0.02,stepsize=0.05):
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
    print("preparations complete")

    # This model has 2 components: Q parameters and switchpoints
    # They are combined in a custom likelihood function

    ###########################################################################
    # Switchpoint:
    ###########################################################################
    # Modeling the movement of the regime shift(s) is the tricky part
    # Regime shifts will only be allowed to happen at a node
    seg_map = tree_map(tree,seglen)
    print("segmap complete")
    switch = [None]*(nregime-1)
    for regime in range(nregime-1):
        switch[regime]= make_switchpoint_stoch(seg_map, name=str("switch_{}".format(regime)))
    print("switchpoints complete")
    ###########################################################################
    # Qparams:
    ###########################################################################
    # Each Q parameter is an exponential
    Qparams = [None] * nparam
    for i in range(nparam):
         Qparams[i] = pymc.Exponential(name=str("Qparam_{}".format(i)), beta=1.0, value=0.1*(i+1))
    print("Qparams complete")


    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function
    if mods is not None:
        l = lf_mk_mr_midbranch_mods(tree=tree, chars=chars,
            mods=mods, pi=pi, findmin=False, orderedparams=False)
    else:
        l = lf_mk_mr_midbranch(tree, chars=chars, Qtype="ARD",
                                               nregime=nregime, pi=pi,
                                               findmin=False)
    print("likelihood function complete")

    @pymc.deterministic
    def likelihood(q = Qparams, s=switch,name="likelihood"):
        return l(q,s)

    @pymc.potential
    def multi_mklik(l=likelihood):
        return l

    if db is None:
        mod = pymc.MCMC(locals())
    else:
        mod = pymc.MCMC(locals(), db=db, dbname=dbname)
    print("model created")
    for s in switch:
        mod.use_step_method(SwitchpointMetropolis, s, tree, seg_map,stepsize=stepsize,seglen=seglen)
    print("step methods used")
    return mod
