"""
Calculate independent contrasts

TODO: include utilities for transforming data, etc.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

def PIC(node, data, results=None):
    """
    Phylogenetic independent contrasts.

    Recursively calculate independent contrasts of a bifurcating node
    given a dictionary of trait values.

    Args:
        node (Node): A node object
        data (dict): Mapping of leaf names to character values

    Returns:
        dict: Mapping of internal nodes to tuples containing ancestral
              state, its variance (error), the contrast, and the
              contrasts's variance.

    TODO: modify to accommodate polytomies.
    """
    #TODO: add option to scale by variance
    if results is None:
        results = {}
    X = []; v = []
    for child in node.children:
        if child.children:
            PIC(child, data, results)
            child_results = results[child]
            X.append(child_results[0])
            v.append(child_results[1])
        else:
            X.append(data[child.label])
            v.append(child.length)

    Xi, Xj = X  # Xi - Xj is the contrast value
    vi, vj = v

    # Xk is the reconstructed state at the node
    Xk = ((1.0/vi)*Xi + (1/vj)*Xj) / (1.0/vi + 1.0/vj)

    # vk is the variance
    vk = node.length + (vi*vj)/(vi+vj)

    results[node] = (Xk, vk, Xi-Xj, vi+vj)

    return results

if __name__ == "__main__":
    from . import tree
    n = tree.read(
        "((((Homo:0.21,Pongo:0.21)N1:0.28,Macaca:0.49)N2:0.13,"\
        "Ateles:0.62)N3:0.38,Galago:1.00)N4:0.0;"
        )
    char1 = {
        "Homo": 4.09434,
        "Pongo": 3.61092,
        "Macaca": 2.37024,
        "Ateles": 2.02815,
        "Galago": -1.46968
        }

    for k, v in list(PIC(n, char1).items()):
        print(k.label or k.id, v)
