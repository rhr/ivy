# Mk multi regime models
from __future__ import absolute_import, division, print_function, unicode_literals

import ivy
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
IDEG = 6 # Constant used in Cython/Fortran code.



def mk_mr_midbranch(tree, chars, Qs, switchpoint, pi="Equal", returnPi=False,
                     ar = None, debug=False):
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

    Returns:
        (float): Log-likelihood of tree/characters given the Q matrices.
    """
    ####################
    # Step 1: setting up
    ####################
    # These lines of code ensure that the object switchpoint is of the correct
    # type even when passed in from PYMC
    if len(switchpoint)>0:
        if type(switchpoint[0]) == pymc.PyMCObjects.Stochastic:
            switchpoint = [i.value for i in switchpoint]

    # chars can be passed in as a dict. If it is, convert it to a list of ints in preorder sequence.
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]

    # Number of character states
    nchar = len(set(chars))

    # Creation of preallocated arrays.
    if ar is None:
        ar = create_mkmr_mb_ar(tree,chars,nregime=Qs.shape[0],findmin=True)
    # chars is going to need to be reordered.
    chars = ar["chars"]
    # Get the actual nodes from the switchpoint tuple
    switchpoint_nodes = [ar["tree_copy"][switchpoint[i][0].id] for i in range(len(switchpoint))]

    # Reset t values
    ar["t"] = ar["blens"][:] + 1e-40 # Branch lengths can't be exactly zero.

    # Adjust t to represent the length of each switchpoint
    for s in switchpoint:
        sw = ar["tree_copy"][s[0].id].pi #Switchpoint
        sk = ar["tree_copy"][s[0].id].parent.pi #Switchpoint's parent
        ar["t"][sw] = s[1]
        ar["t"][sk] =  ar["tree_copy"][s[0].id].parent.length - s[1]


    # inds corresponds to the tree in postorder sequence, with each value
    # corresponding to the regime that node is in. It is passed to dexpm3
    switches_ni = [ar["tree_copy"][s[0].id].ni for s in switchpoint] + [0]
    cyexpokit.inds_from_switchpoint_p(np.array(switches_ni), ar["cladesize_preorder"],
                                    ar["clades_postorder_preorder"],
                                    ar["inds"])
    (Qs != ar["prev_Q"]).any(axis=(1,2), out=ar["Qdif"])

    #####################
    # Step 2: create "pmask"
    #####################
    # pmask is a mask array that keeps track of which p-matrices need to be
    # recalculated and which ones can use the values from the previous call.

    #Which Qs are different? Different Qs mean we have to recalculate p for that index
    np.logical_or.reduce((ar["prev_inds"]!=ar["inds"], ar["t"]!=ar["prev_t"],ar["Qdif"][ar["inds"]]),out=ar["pmask"]) # Which p-matrices do we need to recalcualte?

    ##################
    # Step 3: calculate p matrices
    ##################

    # Creating probability matrices from Q matrices and branch lengths
    # inds indicates which Q matrix to use for which branch
    # np.exp(ar["p"], out=ar["p"]) # Exponentiate p matrices
    Qs += 1e-40
    cyexpokit.lndexpm3(Qs,ar["t"],np.array(ar["inds"]),ar["p"],ideg=IDEG,wsp=ar["wsp"],pmask=ar["pmask"].astype(int)) # Calculating the actual p matrix
    # np.log(ar["p"], out=ar["p"]) # Log p matrices

    #################
    # Step 4: calculate likelihood
    #################

    # Calculating the likelihoods for each node in post-order sequence
    np.copyto(ar["nodelist"], ar["nodelistOrig"]) # Resetting the starting values of nodelist.

    cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"],ar["intnode_list"],ar["child_ar"]) # Performing the likelihood calculation
    # cyexpokit.cy_mk_log(ar["nodelist"], ar["p"], nchar, ar["tmp_ar"],ar["intnode_list"],ar["child_ar"])

    ################
    # Step 5: apply root prior
    ################
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
    ###############
    # Cleanup and storing values for next call to this function
    ##############
    ar["prev_t"][:] = ar["t"][:]
    ar["prev_inds"][:] = ar["inds"][:]
    ar["prev_Q"][:] =  Qs[:]
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

    wsp = np.empty(4*nchar*nchar+IDEG+1)

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
    tree_copy_ = tree.copy()
    for n in tree_copy_.descendants():
        n.meta["cached"]=False
        n.bisect_branch(1e-15, reindex=False)
    # Here we break up the tree so that each node has a "knee"
    # for a parent. This knee starts with a length of 0, effectively
    # making it nonexistant, but can have its length changed
    # to act as a switchpoint.
    tree_copy_str = tree_copy_.write()
    tree_copy = ivy.tree.read(tree_copy_str)
    for n1,n2 in zip(tree_copy_.preiter(), tree_copy.preiter()):
        n2.id = n1.id


    for n in tree_copy:
        n.cladesize = len(n)
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

    pre_post = np.array([n.pi for n in tree_copy]) # The postorder index that corresponds to each preorder index
    prev_inds = np.zeros(len(t), dtype=int) #Keeping track of locations from previous call
    Qdif = np.ones([nregime], dtype=bool) # Array for keeping track of changes to Q
    locs = np.zeros(nregime, dtype=object)
    inds = np.zeros(nt, dtype=int)

    max_children = max(len(n.children) for n in tree_copy)
    child_ar = np.empty([tree_copy.cladesize,max_children], dtype=np.int64)
    child_ar.fill(-1)

    intnode_list = np.array(sorted(set(nodelist[:-1,nchar])), dtype=int)
    for intnode in intnode_list:
        children = np.where(nodelist[:,nchar]==intnode)[0]
        child_ar[int(intnode)][:len(children)] = children

    wsp = np.empty(4*nchar*nchar+IDEG+1)
    cladesize_preorder = np.array([len(n) for n in tree_copy])
    clades_postorder_preorder = np.zeros([len(tree_copy),len(tree_copy)])
    clades_postorder_preorder -= 1
    for ni,node in enumerate(tree_copy):
        clades_postorder_preorder[ni][:cladesize_preorder[ni]] = [x.pi for x in node]

    var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
           "nodelistOrig":nodelistOrig, "upperbound":upperbound,
           "root_priors":rootpriors, "nullval":nullval, "tmp_ar":tmp_ar,
           "tree_copy":tree_copy,"chars":chars,"pre_post":pre_post,"prev_Q":prev_Q,
           "prev_t":prev_t, "blens":blens,"pmask":pmask,
           "prev_inds":prev_inds, "locs":locs,"Qdif":Qdif,
           "intnode_list":intnode_list,"child_ar":child_ar,"wsp":wsp,
           "inds":inds,"cladesize_preorder":cladesize_preorder,
           "clades_postorder_preorder":clades_postorder_preorder}
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
        out = True
        locs = np.zeros(len(switches_c), dtype=object)
    else:
        out = False
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
    locs[:] =locs[[switches_c.index(i) for i in switches]][:]

    if out:
        return locs

def inds_from_switchpoint_cython(tree, switches_ni, ar,debug=False):
    nr = len(switches_ni)-1

    switches_ni_c = switches_ni[:]
    switches_ni_c.sort(key=lambda x: ar["cladesize_preorder"][x])

    switches_key = [switches_ni_c.index(i) for i in switches_ni]

    for i,s in enumerate(switches_ni_c[::-1]):
        for n in ar["clades_postorder_preorder"][s]:
            if not n==len(ar["cladesize_preorder"])-1: # Ignore the root
                ar["inds"][n] = switches_key[nr-i]


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
        root_to_tip_length = tree.max_tippath()
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

        new_location = cyexpokit.local_step(prev_location,self.stepsize,self.seg_size,self.adaptive_scale_factor)

        self.stochastic.value = new_location

    def global_step(self):
        self.stochastic.value = cyexpokit.random_tree_location(self.seg_map)

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
    root_to_tip_length = tree.max_tippath()
    seg_size = seglen * root_to_tip_length
    seg_map = []
    seen = [tree]
    for node in tree.descendants():
        cur_len = node.length
        nseg = int(ceil(cur_len/seg_size))
        for n in range(nseg):
            seg_map.append((node, n*seg_size+1e-15))
    return np.array(seg_map,dtype=object)


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
    startingval = cyexpokit.random_tree_location(seg_map)
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

def mk_multi_bayes(tree, chars,nregime,qidx, pi="Equal" ,seglen=0.02,stepsize=0.05):
    """
    Create a Bayesian multi-mk model. User specifies which regime models
    to use and the Bayesian model finds the switchpoints.

    Args:
        tree (Node): Root node of tree.
        chars (dict): Dict mapping tip labels to discrete character
          states. Character states must be in the form of [0,1,2...]

        regime (int): The number of distinct regimes to test. Set to
          1 for an Mk model, set to greater than 1 for a multi-regime Mk model.
        qidx (np.array): Index specifying the model to test

            columns:
                0, 1, 2 - index axes of q
                3 - index of params
            This scheme allows flexible specification of models. E.g.:
            Symmetric mk2:
                params = [0.2]; qidx = [[0,0,1,0],
                                        [0,1,0,0]]

            Asymmetric mk2:
                params = [0.2,0.6]; qidx = [[0,0,1,0],
                                            [0,1,0,1]]
           NOTE:
             The qidx corresponding to the first q matrix (first column 0)
             is always the root regime
        pi (str or np.array): Option to weight the root node by given values.
           Either a string containing the method or an array
           of weights. Weights should be given in order.

           Accepted methods of weighting root:

           Equal: flat prior
           Equilibrium: Prior equal to stationary distribution
             of Q matrix
           Fitzjohn: Root states weighted by how well they
             explain the data at the tips.
        seglen (float): Size of segments to break tree into. The smaller this
          value, the more "fine-grained" the analysis will be. Optional,
          defaults to 2% of the root-to-tip length.
        stepsize (float): Maximum size of steps for switchpoints to take.
          Optional, defaults to 5% of root-to-tip length.


    """
    if type(chars) == dict:
        data = chars.copy()
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    else:
        data = dict(zip([n.label for n in tree.leaves()],chars))
    # Preparations
    nchar = len(set(chars))
    nparam = len(set([n[-1] for n in qidx]))
    # This model has 2 components: Q parameters and switchpoints
    # They are combined in a custom likelihood function
    ###########################################################################
    # Switchpoint:
    ###########################################################################
    # Modeling the movement of the regime shift(s) is the tricky part
    # Regime shifts will only be allowed to happen at a node
    seg_map = tree_map(tree,seglen)
    switch = [None]*(nregime-1)
    for regime in range(nregime-1):
        switch[regime]= make_switchpoint_stoch(seg_map, name=str("switch_{}".format(regime)))
    ###########################################################################
    # Qparams:
    ###########################################################################
    # Each Q parameter is an exponential
    Qparams = [None] * nparam
    for i in range(nparam):
         Qparams[i] = pymc.Exponential(name=str("Qparam_{}".format(i)), beta=1.0, value=0.1*(i+1))


    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function
    l = cyexpokit.make_mklnl_func(tree, data,nchar,nregime,qidx)

    @pymc.deterministic
    def likelihood(q = Qparams, s=switch,name="likelihood"):
        return l(np.array(q),np.array([x[0].ni for x in s],dtype=np.intp),np.array([x[1] for x in s]))

    @pymc.potential
    def multi_mklik(lnl=likelihood):
        if not (np.isnan(lnl)):
            return lnl
        else:
            return -np.inf
    mod = pymc.MCMC(locals())
    for s in switch:
        mod.use_step_method(SwitchpointMetropolis, s, tree, seg_map,stepsize=stepsize,seglen=seglen)
    return mod
