import ivy
from ivy.chars import mk
import numpy as np
import math
import scipy
import scipy.stats
from scipy.stats import rv_discrete
import random

def sim_discrete(tree, Q, anc=None, charname=None):
    """
    Simulate discrete character on a tree given transition rate matrix.

    Args:
        tree (Node): Root node of tree.
        Q (np.array): Instantaneous rate matrix
        anc (int): Root state. Optional. If None, root state is chosen
          from stationary distrubution of Q matrix
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
        anc = rv_discrete(values=(range(nchar),mk.qsd(Q))).rvs()
    simtree.sim_char = {}
    simtree.sim_char["sim_state"] = anc
    simtree.sim_char["sim_hist"] = []
    ###############################################################
    # Go through the tree in preorder sequence and simulate history
    ###############################################################
    for node in simtree.descendants():
        prevstate = node.parent.sim_char["sim_state"]
        node.sim_char = {}
        node.sim_char["sim_state"] = prevstate
        node.sim_char["sim_hist"] = []
        if node.length == 0:
            pass
        else:
            dt = 0
            while True:
                dt += np.random.exponential(-Q[prevstate,prevstate])
                if dt > node.length:
                    break
                newstate = rv_discrete(values=(range(nchar)[:prevstate] + range(nchar)[prevstate+1:],
                                        np.concatenate((Q[prevstate][:prevstate],Q[prevstate][prevstate+1:])))).rvs()
                node.sim_char["sim_hist"].append((newstate,dt))
                node.sim_char["sim_state"] = newstate
                prevstate = newstate
    return simtree
