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

t = mk_multi_regime(tree, chars, Qs, locs, pi=pi)

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

from ivy.interactive import *
%pylab
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

# The Q-matrices used to generate the character states
trueFastQ = np.array([[-.8,.8], [.8,-.8]])
trueSlowQ = np.array([[-.1,.1], [.1,-.1]])
trueQs = np.array([trueFastQ,trueSlowQ])

mods = [(2,2),(1,1)]

mod_mr = mk_multi_bayes(tree, mr_chars, mods=mods,nregime=2, orderedparams=False)
mod_mr.sample(10000,burn=1000,thin=3)

plt.plot(mod_mr.trace("switch_0")[:])

plt.plot(mod_mr.trace(str("Qparam_0"))[:])
plt.plot(mod_mr.trace(str("Qparam_1"))[:])


nds = [tree[int(i)] for i in mod_mr.trace(str("switch_0"))[:]]
fig = treefig(tree)
fig.toggle_branchlabels()
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tipstates(chars)


mod = mk_multi_bayes(tree, mr_chars, mods=[(1,1),(2,2)])
mod.sample(10000,burn=1000,thin=3)

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
# Q1 = np.array([[-1e-15,1e-15],
#                 [1e-15,-1e-15]])
# Q2 = np.array([[-0.05,0.05],
#                [0.05,-0.05]])
# Q3 = np.array([[-1.,1.],
#                [1.,-1.]])
#
# true_Qs = np.array([Q2,Q3,Q1])
#
# true_l = mk_mr.mk_multi_regime(tree, chars_r3, true_Qs, true_locs)
#-212.46280532572879
mods = [(3, 3), (1, 1), (2, 2)]

mod_r3 = mk_multi_bayes(tree, chars, mods=mods,orderedparams=False)

mod_r3.sample(20000)

plt.plot(mod_r3.trace(str("switch_1"))[:])
plt.plot(mod_r3.trace(str("switch_0"))[:])


mods = [(2, 2), (3, 3), (1, 1)]

mod_r3 = mk_mr.mk_multi_bayes(tree, chars_r3, mods=mods)

mod_r3.sample(20000, burn=2000, thin=3)
