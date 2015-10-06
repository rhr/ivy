"""
Likelihood calculation of discrete traits
"""
import ivy
import numpy as np
from ivy.chars.expokit import cyexpokit
from scipy.optimize import minimize

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

def treeLikelihood(tree, chars, Q):
    """
    Calculate likelihood for the root node of a tree given a list of characters
    and a Q matrix

    Args:
        tree (Node): Root node of a tree. All branch lengths must be greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in preoder sequence. Character states must be numbered 0,1,2,...
        Q (np.array): Instantaneous rate matrix
    """
    chartree = tree.copy()
    chartree.char = None; chartree.likelihoodNode={}
    t = [node.length for node in chartree.descendants()]
    t = np.array(t, dtype=np.double)

    p = cyexpokit.dexpm_tree(Q, t)

    for i, nd in enumerate(chartree.descendants()):
        nd.pmat = p[i]
        nd.likelihoodNode = {}
        nd.char = None

    for i, lf in enumerate(chartree.leaves()):
        lf.char = chars[i]


    for node in chartree.postiter():
        if node.char is not None: # For tip nodes, likelihoods are 1 for observed state and 0 for the rest
            for state in range(Q.shape[0]):
                node.likelihoodNode[state]=0.0
            node.likelihoodNode[node.char]=1.0
        else:
            for state in range(Q.shape[0]):
                likelihoodStateN = []
                for ch in node.children:
                    likelihoodStateNCh = []
                    for chState in range(Q.shape[0]):
                        likelihoodStateNCh.append(ch.pmat[state, chState] * ch.likelihoodNode[chState])
                    likelihoodStateN.append(sum(likelihoodStateNCh))
                node.likelihoodNode[state]=np.product(likelihoodStateN)
    return sum(chartree.likelihoodNode.values())


def estimateQBayesian(tree, chars):
    """
    Estimate maximum likelihood value and posterior probability distribution
    of Q

    Following Pagel 2004:

    p(Qi | D, T) = (p(D|Qi) * p(Qi))/integralQ(p(D|Q)p(Q)d(Q))
    """
    pass

def create_likelihood_function_oneparam(tree, chars):
    """
    Create a function that takes values for Q and returns likelihood.

    Returned function to be passed into scipy.optimize
    """
    def likelihood_function(Qparam):
        Q = np.array([[0-Qparam,Qparam],[Qparam, 0-Qparam]], dtype=np.double)
        return treeLikelihood(tree, chars, Q)
    return likelihood_function

def estimateQLikelihood_oneparam(tree, chars):
    """
    Estimate maximum likelihood values of Q

    One-parameter model: alpha = beta
    """
