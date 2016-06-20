import cyexpokit
import ivy
import numpy as np

qidx = np.array(
    [[0,0,1],
     [0,1,0]],
    dtype=np.intp)

## r = ivy.tree.read("(A:1,B:1)root;")
## data = dict(A=0,B=1)
## f = cyexpokit.make_mklnl_func(r, data, 2, 1, qidx)
## print f(np.array([0.1,0.1]))

rand100 = ivy.tree.read("../../tests/support/randtree100tips.newick")
charv = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

labels = [ lf.label for lf in rand100.leaves() ]
data = dict(zip(labels, charv))

f = cyexpokit.make_mklnl_func(rand100, data, 2, 1, qidx)
print f(np.array([0.4549581, 0.4549581]))
