#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import math

import numpy as np

import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import catpars

"""
Ancestor state reconstruction
"""

def anc_recon_cat(tree, chars, Q, p=None, pi="Equal", ars=None, nregime=1):
    """
    Given tree, character states at tips, and transition matrix perform
    ancestor reconstruction for a discrete character.

    Perform downpass using mk function, then perform uppass.

    Return reconstructed states - including tips

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        Q (np.array): Instantaneous rate matrix
        p (np.array): 3-D array of dimensions branch_number * nchar * nchar.
            Optional. Pre-allocated space for efficient calculations
        pi (str or np.array): Option to weight the root node by given values.
           Either a string containing the method or an array
           of weights. Weights should be given in order.
           Accepted methods of weighting root:
             Equal: Flat prior
             Equilibrium: Prior equal to stationary distribution
               of Q matrix
             Fitzjohn: Root states weighted by how well they
               explain the data at the tips.
        ars (dict): Dict of pre-allocated arrays to improve
          speed by avoiding creating and destroying new arrays. Can be
          created with create_ancrecon_ars function. Optional.
        nregime (int): If hidden-rates model is used, the number
          of regimes to consider in the reconstruction. Optional, defaults
          to 1 (no hidden rates)
    Returns:
        np.array: Array of nodes in preorder sequence containing marginal
          likelihoods.
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = Q.shape[0]
    if ars is None:
        # Creating arrays to be used later
        ars = create_ancrecon_ars(tree, chars, nregime)
    # Calculating the likelihoods for each node in post-order sequence
    if p is None: # Instantiating empty array
        p = np.empty([len(ars["t"]), Q.shape[0],
                     Q.shape[1]], dtype = np.double, order="C")
    # Creating probability matrices from Q matrix and branch lengths
    cyexpokit.dexpm_tree_preallocated_p(Q, ars["t"], p) # This changes p in place
    np.copyto(ars["down_nl_w"], ars["down_nl_r"]) # Copy original values if they have been changed
    ars["child_inds"].fill(0)
    root_equil = ivy.chars.mk.qsd(Q)

    cyexpokit.cy_anc_recon(p, ars["down_nl_w"], ars["charlist"], ars["childlist"],
                        ars["up_nl"], ars["marginal_nl"], ars["partial_parent_nl"],
                        ars["partial_nl"], ars["child_inds"], root_equil,ars["temp_dotprod"],
                        nregime)

    return ars["marginal_nl"]

def anc_recon_mkmr(root,chars,Q,switchpoint):
    nregime = Q.shape[0]
    root_copy = root.copy()
    for node in root_copy.descendants():
        node.meta["cached"]=False
        node.bisect_branch(1e-15,reindex=False)
    root_copy.reindex()
    root_copy.set_iternode_cache()
    for node in root_copy.iternodes():
        node.meta["cached"] = True
    nchar = len(set(chars))
    ars = create_ancrecon_ars(tree, chars, nregime)

def create_ancrecon_ars(tree, chars, nregime = 1):
    """
    Create nodelists. For use in recon function

    Returns edgelist of nodes in postorder sequence, edgelist of nodes in preorder sequence,
    partial likelihood vector for each node's children, childlist, and branch lengths.
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)
    nobschar = len(set(chars))
    nchar = nobschar * nregime
    preleaves = [ n for n in tree.preiter() if n.isleaf ]
    postleaves = [n for n in tree.postiter() if n.isleaf ]
    postnodes = list(tree.postiter())
    prenodes = list(tree.preiter())
    postChars = [ chars[i] for i in [ preleaves.index(n) for n in postleaves ] ]
    nnode = len(t)+1
    down_nl = np.zeros([nnode, (nchar)+1])
    up_nl = np.zeros([nnode, (nchar)+2])
    childlist = np.zeros(nnode, dtype=object)
    leafind = [ n.isleaf for n in tree.postiter()]

    # --------- Down nodelist(nl), postorder sequence
    for k,ch in enumerate(postChars):
        # Setting initial tip likelihoods to 1 or 0
        # TODO: Switch to log calculations
        for r in range(nregime):
            [ n for i,n in enumerate(down_nl) if leafind[i] ][k][ch+(r*nobschar)] = 1.0
    for i,n in enumerate(down_nl[:-1]):
        n[nchar] = postnodes.index(postnodes[i].parent)
        childlist[i] = [ nod.pi for nod in postnodes[i].children ]
    childlist[i+1] = [ nod.pi for nod in postnodes[i+1].children ]
    # Setting initial node likelihoods to one for calculations
    down_nl[[ i for i,b in enumerate(leafind) if not b],:-1] = 1.0

    # -------- Up nl, preorder sequence
    up_nl[:,:-1].fill(1)
    for i,n in enumerate(up_nl[1:]):
        n[nchar] = postnodes.index(prenodes[1:][i].parent)
        n[nchar+1] = postnodes.index(prenodes[1:][i])
    # -------- Marginal nl, also preorder sequence
    marginal_nl = up_nl.copy()

    # ------- Partial likelihood vectors
    partial_nl = np.zeros([nnode,max([n.nchildren for n in tree]),nchar])

    # ------- Parent partial likelihood vectors
    partial_parent_nl = np.zeros([up_nl.shape[0],nchar])

    # ------- Child masking arrays
    child_inds = np.zeros(partial_nl.shape[0], dtype=int)

    # ------- Character list
    charlist = list(range(nchar))

    # ------- Empty array to store root priors
    root_priors = np.empty([nchar], dtype=np.double)

    # ------- Temporary array to store dot products
    temp_dotprod = np.empty([nchar], dtype=np.double)

    names = ["down_nl_r","down_nl_w","up_nl","marginal_nl",
            "partial_nl","partial_parent_nl","child_inds",
            "childlist","t","charlist","root_priors","temp_dotprod"]
    ar_list = [down_nl,down_nl.copy(),up_nl,marginal_nl,partial_nl,partial_parent_nl,child_inds,
           childlist,t,charlist,root_priors,temp_dotprod]
    ar_dict = {name:ar for name,ar in zip(names, ar_list)}

    return ar_dict

def parsimony_recon(tree, chars):
    """
    Use parsimony to reconstruct the minimum number of changes needed to observe
    states at the tips.

    Equal weights assumed

    Args:
        tree (Node): Root node of tree
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
    Returns:
        np.array: Array of assigned states of interior nodes
    """
    stepmatrix = catpars.default_costmatrix(len(set(chars.values())))
    rec = catpars.ancstates(tree, chars, stepmatrix)
    for k in list(rec.keys()):
        rec[k] = rec[k][0]
    return rec


def pscore(tree, chars):
    """
    Return the minimum number of changes needed by parsimony to observe
    given data on the tree

    Args:
        tree (Node): Root node of the tree
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...
    """
    if len(set(chars)) == 1:
        return 0
    precon =  parsimony_recon(tree, chars)
    minp = 0.0
    chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    for node in tree:
        if not node.isleaf:
            ch = get_childstates(node, precon, chars)
            pa = precon[node]
            minp += average_nchanges(ch, pa)
    return minp


def get_childstates(node, precon, chars):
    """
    For use in pscore function
    """
    chstates = [None] * len(node.children)
    for i,n in enumerate(node.children):
        if n.isleaf:
            chstates[i] = chars[n.li]
        else:
            chstates[i] = precon[n]
        if not hasattr(chstates[i], "__iter__"):
            chstates[i] = [chstates[i]]
    return chstates


def average_nchanges(ch, pa):
    """
    For use in pscore function
    """
    lp = 1.0/len(pa)
    nchanges = 0
    for p in pa:
        for c in ch:
            lc = 1.0/len(c)
            for s in c:
                if s != p:
                    nchanges += lp * lc
    return nchanges
