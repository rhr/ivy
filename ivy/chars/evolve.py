#!/usr/bin/env python
"""
Functions for evolving traits and trees.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
def brownian(root, sigma=1.0, init=0.0, values={}):
    """
    Recursively evolve a trait by Brownian motion up from the node
    *root*.

    * *sigma*: standard deviation of the normal random variate after
      one unit of branch length

    * *init*: initial value

    Returns: *values* - a dictionary mapping nodes to evolved values
    """
    from scipy.stats import norm
    values[root] = init
    for child in root.children:
        time = child.length
        random_step = norm.rvs(init, scale=sigma*time)
        brownian(child, sigma, random_step, values)
    return values

def test_brownian():
    """
    Evolve a trait up an example tree of primates:.

    ((((Homo:0.21,Pongo:0.21)N1:0.28,Macaca:0.49)N2:0.13,
    Ateles:0.62)N3:0.38,Galago:1.00)root;

    Returns: (*root*, *data*) - the root node and evolved data.
    """
    import newick
    root = newick.parse(
        "((((Homo:0.21,Pongo:0.21)N1:0.28,Macaca:0.49)N2:0.13,"\
        "Ateles:0.62)N3:0.38,Galago:1.00)root;"
        )
    print(root.ascii(scaled=True)) 
    evolved = brownian(root)
    for node in root.iternodes():
        print(node.label, evolved[node])
    return root, evolved

if __name__ == "__main__":
    test_brownian()
