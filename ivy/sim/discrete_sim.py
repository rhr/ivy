from __future__ import absolute_import, division, print_function, unicode_literals

import ivy
from ivy.chars import mk
import numpy as np
import math
import scipy
import scipy.stats
from scipy.stats import rv_discrete
import random

def sim_discrete(tree, Q, anc=None, charname=None, rseed=None):
    """
    Simulate discrete character on a tree given transition rate matrix.

    Args:
        tree (Node): Root node of tree.
        Q (np.array): Instantaneous rate matrix
        anc (int): Root state. Optional. If None, root state is chosen
          from stationary distrubution of Q matrix
        rseed (int): Random seed to be passed to np.random. Optional
    Returns:
        Node: Copy of tree with each node containing its simulated
          state and the character history of its subtending branch.
    """
    assert tree.isroot, "Given node must be root"
    nchar = Q.shape[0]
    simtree = tree.copy()


    ####################################
    # Start with ancestral state of root
    ####################################
    if anc is None:
        # Randomly pick ancestral state from stationary distribution
        anc = rv_discrete(values=(list(range(nchar)),mk.qsd(Q))).rvs()
    simtree.sim_char = {}
    simtree.sim_char["sim_state"] = anc
    simtree.sim_char["sim_hist"] = []
    ###############################################################
    # Go through the tree in preorder sequence and simulate history
    ###############################################################
    if rseed:
        np.random.seed(rseed)
    for node in simtree.descendants():
        prevstate = node.parent.sim_char["sim_state"]
        node.sim_char = {}
        node.sim_char["sim_state"] = prevstate
        node.sim_char["sim_hist"] = []
        if node.length == 0:
            pass
        else:
            dt = 0
            while 1:
                dt += np.random.exponential(1.0/-Q[prevstate,prevstate])
                if dt > node.length:
                    break
                vals  = np.concatenate((Q[prevstate][:prevstate],Q[prevstate][prevstate+1:]))
                newstate = rv_discrete(values=(list(range(nchar))[:prevstate] + list(range(nchar))[prevstate+1:],
                                    vals/sum(vals))).rvs()
                node.sim_char["sim_hist"].append((newstate,dt))
                node.sim_char["sim_state"] = newstate
                prevstate = newstate
    return simtree
#
# def sim_discrete_nielsen(tree, Q, anc=None, charname=None):
#     """
#     Simulate discrete character on a tree given transition rate matrix
#     following Nielsen 2001.
#
#     Args:
#         tree (Node): Root node of tree.
#         Q (np.array): Instantaneous rate matrix
#         anc (int): Root state. Optional. If None, root state is chosen
#           from stationary distrubution of Q matrix
#     Returns:
#         Node: Copy of tree with each node containing its simulated
#           state and the character history of its subtending branch.
#           Simulated states contained in the attribute "sim_char"
#     """
#     assert tree.isroot, "Given node must be root"
#     nchar = Q.shape[0]
#     simtree = tree.copy()
#
#
#     ####################################
#     # Start with ancestral state of root
#     ####################################
#     if anc is None:
#         # Randomly pick ancestral state from stationary distribution
#         anc = rv_discrete(values=(range(nchar),mk.qsd(Q))).rvs()
#     simtree.sim_char = {}
#     simtree.sim_char["sim_state"] = anc
#     simtree.sim_char["sim_hist"] = []
#     ###############################################################
#     # Go through the tree in preorder sequence and simulate history
#     ###############################################################
#     for node in simtree.descendants():
#         prevstate = node.parent.sim_char["sim_state"]
#         node.sim_char = {}
#         node.sim_char["sim_hist"] = []
#         QA = -1*Q[prevstate,prevstate]
#         if node.length == 0:
#             pass
#         else:
#             stateprobs = scipy.linalg.expm(Q * node.length)[prevstate]
#             newstate = rv_discrete(values=(range(nchar),stateprobs)).rvs()
#             node.sim_char["sim_state"] = newstate
#             if newstate == prevstate:
#                 nmuts = np.random.poisson(QA*node.length)
#             else:
#                 nmuts = np.random.poisson(QA*node.length)
#                 t1 = -np.log(1 - random.random() * math.exp(-QA*node.length))/(-1*QA)
#
#     return simtree
