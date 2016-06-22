import newick, cy_tree
import numpy as np

s = '((((Homo:0.21,Pongo:0.21)A:0.28,Macaca:0.49)B:0.13,Ateles:0.62)C:0.38,Galago:1.00)root;'
r = newick.parse(s)
t = cy_tree.Tree(r)
a = np.empty(t.nnodes, dtype=int)
cy_tree.cladesizes3(t, a)
print a
