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
import random

from ivy.chars.mk import *
from ivy.chars.mk_mr import *
from ivy.chars.hrm import *
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
