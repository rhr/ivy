#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import scipy
import numpy


def default_costmatrix(numstates, dtype=numpy.int):
    "a square array with zeroes along the diagonal, ones elsewhere"
    return scipy.logical_not(scipy.identity(numstates)).astype(float)


def minstates(v):
    "return the indices of v that equal the minimum"
    return scipy.nonzero(scipy.equal(v, min(v)))


def downpass(node, states, stepmatrix, chardata, node2dpv=None):
    if node2dpv is None:
        node2dpv = {}

    if not node.isleaf:
        for child in node.children:
            downpass(child, states, stepmatrix, chardata, node2dpv)

        dpv = scipy.zeros([len(states)])
        node2dpv[node] = dpv
        for i in states:
            for child in node.children:
                child_dpv = node2dpv[child]
                mincost = min([ child_dpv[j] + stepmatrix[i,j] \
                                for j in states ])
                dpv[i] += mincost

        #print node.label, node.dpv

    else:
        #print node.label, chardata[node.label]
        node2dpv[node] = stepmatrix[:,chardata[node.label]]

    return node2dpv


def uppass(node, states, stepmatrix, node2dpv, node2upm={},
           node2ancstates=None):
    parent = node.parent
    if not node.isleaf:
        if parent is None: # root
            dpv = node2dpv[node]
            upm = None
            node.mincost = min(dpv)
            node2ancstates = {node: minstates(dpv)}

        else:
            M = scipy.zeros(stepmatrix.shape)
            for i in states:
                sibs = [ c for c in parent.children if c is not node ]
                for j in states:
                    c = 0
                    for sib in sibs:
                        sibdpv = node2dpv[sib]
                        c += min([ sibdpv[x] + stepmatrix[j,x]
                                   for x in states ])
                    c += stepmatrix[j,i]

                    p_upm = node2upm.get(parent)
                    if p_upm is not None:
                        c += min(p_upm[j])

                    M[i,j] += c

            node2upm[node] = M

            v = node2dpv[node][:]
            for s in states:
                v[s] += min(M[s])
            node2ancstates[node] = minstates(v)

        for child in node.children:
            uppass(child, states, stepmatrix, node2dpv, node2upm,
                   node2ancstates)

    return node2ancstates


def ancstates(tree, chardata, stepmatrix):
    """
    Return parsimony ancestral states

    Args:
        tree (Node): Root of tree
        chardata (Dict): Dict of tip labels mapping to character states
        stepmatrix (np.array): Step matrix. Create default step matrix
          with default_costmatrix().
    Returns:
        dict: Internal nodes mapping to reconstructed states.
    """
    states = list(range(len(stepmatrix)))
    return uppass(tree, states, stepmatrix,
                  downpass(tree, states, stepmatrix, chardata))


def _bindeltran(node, stepmatrix, node2dpv, node2deltr=None, ancstate=None):
    if node2deltr is None:
        node2deltr = {}

    dpv = node2dpv[node]
    if ancstate is not None:
        c, s = min([ (cost+stepmatrix[ancstate,i], i) \
                     for i, cost in enumerate(dpv) ])
    else:
        c, s = min([ (cost, i) for i, cost in enumerate(dpv) ])

    node2deltr[node] = s
    for child in node.children:
        _bindeltran(child, stepmatrix, node2dpv, node2deltr, s)

    return node2deltr


def binary_deltran(tree, chardata, stepmatrix):
    states = list(range(len(stepmatrix)))
    node2dpv = downpass(tree, states, stepmatrix, chardata)
    node2deltr = _bindeltran(tree, stepmatrix, node2dpv)
    return node2deltr


if __name__ == "__main__":
    from pprint import pprint
    from ivy import tree
    root = tree.read("(a,((b,c),(d,(e,f))));")

    nstates = 4
    states = list(range(nstates))
    cm = default_costmatrix(nstates)
    chardata = dict(list(zip("abcdef", list(map(int, "000233")))))
    dp = downpass(root, states, cm, chardata)

    for i, node in enumerate(root):
        if not node.label:
            node.label = "N%s" % i
        else:
            node.label = "%s (%s)" % (node.label, chardata[node.label])

    print(ascii.render(root))


##     nstates = 2
##     leaves = tree.leaves()
##     for leaf in leaves:
##         leaf.anc_cost_vector = chardata[leaf.label]

    pprint(
        #ancstates(root, chardata, cm)
        #uppass(root, states, cm, downpass(tree, states, cm, chardata))
        dp
        )
