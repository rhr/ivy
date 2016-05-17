import ivy
from ivy.chars import discrete
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


mod = mk_multi_bayes(tree, mr_chars, mods=[(1,1),(2,2)],switch_step="rand")
mod.sample(5000,burn=500,thin=3)

switch = [int(i) for i in mod.trace("switch")[:]]


nds = [tree[int(i)] for i in mod.trace("switch")[:]]
fig = treefig(tree)
fig.toggle_branchlabels()
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tip_chars(chars)


mod = mk_multi_bayes(tree, mr_chars, mods=[(1,1),(2,2)],switch_step="adj")
mod.sample(10000,burn=1000,thin=3)

switch = [int(i) for i in mod.trace("switch")[:]]


nds = [tree[int(i)] for i in mod.trace("switch")[:]]
fig = treefig(tree)
fig.toggle_branchlabels()
fig.add_layer(ivy.vis.layers.add_node_heatmap,nds, store="hm")
fig.tip_chars(chars)
