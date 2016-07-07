import ivy
from ivy.chars import mk_mr
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt

from ivy.interactive import *
from ivy.vis import layers
tree = ivy.tree.read("support/hrm_600tips.newick")

chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# fig = treefig(tree)
# fig.tipstates(chars_r3)
#
# true_locs = mk_mr.locs_from_switchpoint(tree, [tree[579], tree[329]])
#
Q1 = np.array([[-1e-15,1e-15],
                [1e-15,-1e-15]])
Q2 = np.array([[-0.05,0.05],
               [0.05,-0.05]])
Q3 = np.array([[-1.,1.],
               [1.,-1.]])

true_Qs = np.array([Q2,Q3,Q1])

switchpoints = [(tree[579],0.0),(tree[329],0.0)]

# ar = mk_mr.create_mkmr_mb_ar(tree, chars, 3)
#
# %timeit mk_mr.mk_mr_midbranch(tree, chars, true_Qs, switchpoints,ar=ar)

# true_l = mk_mr.mk_mr(tree, chars_r3, true_Qs, true_locs)
#-212.46280532572879
mods = [(3, 3), (1, 1), (2, 2)]
qidx = np.array(
    [[0,0,1,0],
     [0,1,0,0],
     [1,0,1,1],
     [1,1,0,1],
     [2,0,1,2],
     [2,1,0,2]],
    dtype=np.intp)
import line_profiler
from ivy.chars.expokit import cyexpokit


data = dict(zip([n.label for n in tree.leaves()],chars))
my_f = cyexpokit.make_mklnl_func(tree, data, 2, 3, qidx=qidx)

profile = line_profiler.LineProfiler(my_f)
params = np.array([.5,.6,.7])
switches = np.array([200, 300])
lengths = np.array([.3,1.2])
profile.run("my_f(params,switches,lengths)")
profile.print_stats()

################
# MCMC timeit
################

qidx = np.array(
    [[0,0,1,0],
     [0,1,0,0],
     [1,0,1,1],
     [1,1,0,1],
     [2,0,1,2],
     [2,1,0,2]],
    dtype=np.intp)
mod_r3 = mk_mr.mk_multi_bayes(tree, chars, 3,qidx=qidx)

mod_r3.sample(2000)
