from __future__ import absolute_import, division, print_function, unicode_literals
import ivy
from ivy.chars import mk_mr
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt

from ivy.interactive import *


tree = ivy.tree.read("support/hrm_600tips.newick")
Q = np.array([[-0.02,0.01,0.01,0],
              [0.01,-.02,0,0.01],
              [0.01,0,-.11,0.1],
              [0,0.01,0.1,-.11]])
sim = ivy.sim.discrete_sim.sim_discrete(tree, Q)
fig = treefig(sim)
fig.add_layer(ivy.vis.layers.add_branchstates)

# Plan: use mcmc to test for a specific type
# of regime shift (specify models of both regimes)

Qs = np.array([[[-0.1,0.1],
               [0.1,-0.1]],
               [[-0.5,0.5],
                [0.5,-0.5]]])

locs = locs_from_switchpoint(tree, tree[29])
pi = "Equal"

t = mk_mr(tree, chars, Qs, locs, pi=pi)

f = create_likelihood_function_multimk(tree, chars, "ARD", 2)
Qparams = np.array([[ 0.1,  0.1],
                [ 0.5,  0.5]])


f(Qparams, locs)

def tipchars(sim, mod=4):
    return [n.sim_char["sim_state"]%mod for n in sim.leaves()]
Q = np.array([[-0.02,0.00,0.01,0],
              [0.00,-.02,0,0.01],
              [0.01,0,-.11,0.5],
              [0,0.01,0.5,-.11]])
for i in range(4,100):
    sim = ivy.sim.discrete_sim.sim_discrete(tree, Q, rseed=i)
    if len(set(tipchars(sim))) == 4:
        break


sfig = treefig(sim)
sfig.add_layer(ivy.vis.layers.add_branchstates)


chars = tipchars(sim,2)
o = sfig.selected
l = [x.li for x in o if x.isleaf]
chars = [0 if i in l else c for i,c in enumerate(chars)]

mod = mk_multi_bayes(tree, chars, mods=[(1,1),(2,2)],switch_step="adj")
mod.sample(10000,burn=1000,thin=3)

sws = [int(i) for i in mod.trace("switch")[:]]
nds = [tree[int(i)] for i in mod.trace("switch")[:]]

fig = treefig(tree)
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tip_chars(chars)



QA =  [x for x in mod.trace("Qparam0")[:]]
QB =  [x for x in mod.trace("Qparam1")[:]]



####################################
# Tests
####################################
###################
# Two regime
###################
import ivy
from ivy.chars import mk_mr
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.expokit import cyexpokit

from ivy.interactive import *
tree = ivy.tree.read("support/Mk_two_regime_tree.newick")


mr_chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0]
data = dict(zip([n.label for n in tree.leaves()],mr_chars))
# The Q-matrices used to generate the character states
trueFastQ = np.array([[-.8,.8], [.8,-.8]])
trueSlowQ = np.array([[-.1,.1], [.1,-.1]])
trueQs = np.array([trueFastQ,trueSlowQ])
ar = mk_mr.create_mkmr_mb_ar(tree, mr_chars, 2)

switchpoint =[(tree[350],0.0)]

mods = [(2,2),(1,1)]

qidx = np.array([[0,0,1,0],
                 [0,1,0,0],
                 [1,0,1,1],
                 [1,1,0,1]])
switches = np.array([120])
lengths = np.array([1e-15])
f = cyexpokit.make_mklnl_func(tree, data, 2, 2, qidx)
f(np.array([0.1,0.05]),switches,lengths)


mod_mr = mk_mr.mk_multi_bayes(tree, data,nregime=2,qidx=qidx,stepsize=0.2)
mod_mr.sample(10000,burn=1000,thin=3)


chars = mr_chars
pi = "Equal"
nregime =2
db = None
dbname = None
orderedparams = True
seglen = 0.02
stepsize = 0.2

l = lf_mk_mr_midbranch_mods(tree=tree, chars=chars,
    mods=mods, pi=pi, findmin=False, orderedparams=orderedparams)

seg_map = tree_map(tree,seglen)

switchpoint = [seg_map[500]]
Qparams = [0.1, 1.0]

chardict = {tree.leaves()[i].label:v for i,v in enumerate(chars)}
ar = create_mkmr_mb_ar(tree, chardict, nregime, findmin=True)


cProfile.run("mk_mr_midbranch(tree,chars,trueQs,switchpoint,ar=ar)")

x = mod_mr.trace(str("switch_0"))[:]
fig = treefig(tree)
fig.add_layer(ivy.vis.layers.add_tree_heatmap, x)

plt.plot([n[0].ni for n in x])

plt.plot(mod_mr.trace(str("Qparam_0"))[:])
plt.plot(mod_mr.trace(str("Qparam_1"))[:])


nds = [tree[int(i)] for i in mod_mr.trace(str("switch_0"))[:]]
fig = treefig(tree)
fig.toggle_branchlabels()
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tipstates(chars)


mod = mk_multi_bayes(tree, mr_chars, mods=[(1,1),(2,2)])
mod.sample(1000,burn=1000,thin=3)

switch = [int(i) for i in mod.trace("switch")[:]]


nds = [tree[int(i)] for i in mod.trace("switch")[:]]
fig = treefig(tree)
fig.toggle_branchlabels()
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tip_chars(chars)

#############################################################
# Three-regime chars
#############################################################
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
mod_r3 = mk_mr.mk_multi_bayes(tree, chars, 3,qidx=qidx)

mod_r3.sample(2000)

qp1 = mod_r3.trace("Qparam_0")[150]
qp2 = mod_r3.trace("Qparam_1")[150]
qp3 = mod_r3.trace("Qparam_2")[150]


eQ1 = np.array([[-qp1,qp1],
                [qp1,-qp1]])
eQ2 = np.array([[-qp2,qp2],
               [qp2,-qp2]])
eQ3 = np.array([[-qp3,qp3],
               [qp3,-qp3]])

est_Qs = np.array([eQ2,eQ3,eQ1])

s1 = mod_r3.trace("switch_0")[150]
s2 = mod_r3.trace("switch_1")[150]

mk_mr.mk_mr_midbranch(tree, chars, est_Qs, [s1,s2])

l = mod_r3.trace("likelihood")[150]

x0 = mod_r3.trace(str("switch_0")[:])
fig = treefig(tree)
fig.add_layer(ivy.vis.layers.add_tree_heatmap, x0, store="switch0")

x1 = mod_r3.trace(str("switch_1")[:])
fig.add_layer(ivy.vis.layers.add_tree_heatmap, x0, store="switch1",color="blue")


s0 = [s[0].ni for s in x0]
s1 = [s[0].ni for s in x1]

plt.plot(s0)
plt.plot(s1)


mods = [(2, 2), (3, 3), (1, 1)]

mod_r3 = mk_mr.mk_multi_bayes(tree, chars_r3, mods=mods)

mod_r3.sample(20000, burn=2000, thin=3)


#############################################
# Nodesteps on branches
#############################################

tree = ivy.tree.read(u"((A:1,B:1)C:1,D:2)root;")
Q1 = np.array([[-0.05,0.05],
               [0.05,-0.05]])
Q2 = np.array([[-1.,1.],
               [1.,-1.]])

Qs = np.array([Q1,Q2])

#######################################
# Visualization
#######################################
import ivy
from ivy.chars import mk_mr
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
import pickle

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
dat = pickle.load(open("/home/cziegler/Dropbox/multiregime-discrete/BAMM-like_bayesian/threeregime_bammlike/threeregime_mkmr.p","rb"))

switch_0 = dat["switch_0"]
