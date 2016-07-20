"""
Compute lineages through time
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy


def traverse(node, t=0, results=None):
    """
    Recursively traverse the tree and collect information about when
    nodes split and how many lineages are added by its splitting.
    """
    if results is None:
        results = []
    if node.children:
        ## if not node.label:
        ##     node.label = str(node.id)
        results.append((t, len(node.children)-1))
        for child in node.children:
            traverse(child, t+child.length, results)
    return results

def ltt(node):
    """
    Calculate lineages through time.  The tree is assumed to be an
    ultrametric chronogram (extant leaves, with branch lengths
    proportional to time).

    Args:
        node (Node): A node object. All nodes should have branch lengths.

    Returns:
        tuple: (times, diversity) - 1D-arrays containing the results.
    """
    v = traverse(node) # v is a list of (time, diversity) values
    v.sort()
    # for plotting, it is easiest if x and y values are in separate
    # sequences, so we create a transposed array from v
    times, diversity = numpy.array(v).transpose()
    return times, diversity.cumsum()

def test():
    from . import newick, ascii
    n = newick.parse("(((a:1,b:2):3,(c:3,d:1):1,(e:0.5,f:3):2.5):1,g:4);")
    v = ltt(n)
    print(ascii.render(n, scaled=1))
    for t, n in v:
        print(t, n)

if __name__ == "__main__":
    test()
