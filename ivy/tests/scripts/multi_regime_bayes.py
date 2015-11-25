# Goal: find the MAP estimate for Q for a one-rate, two-state mk model using emcee
from ivy import vis
import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
from ivy.chars import discrete
from ivy.chars import bayesian_models
from matplotlib import pyplot as plt
import pymc
from ivy.interactive import *
from ivy.vis import layers

#
# # slow_tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/Mk_slow_regime_tree.newick")
# fast_tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/Mk_fast_regime_tree.newick")
# fast_tree.length = 0.005
# fast_tree.children[0].length -= .005
# fast_tree.children[1].length -=.005
#
# mr_tree = slow_tree.copy()
#
# x = (0.500 - slow_tree["s144"].rootpath_length(slow_tree["518"]))/slow_tree["518"].length
#
#
# switch_node = mr_tree["518"].bisect_branch(x)
#
#
# switch_node.add_child(fast_tree)
# mr_tree["t99"].rootpath_length()

#
# # multi-regime tree
mr_tree = ivy.tree.read("support/Mk_two_regime_tree.newick")

tree = mr_tree

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
chars = mr_chars

sr_mrmodel = bayesian_models.create_multi_mk_model(tree, chars, Qtype="ER", pi="Fitzjohn", nregime=2)

# fig = treefig(mr_tree)
# cols = map(lambda x: "black" if x==0 else "red", mr_chars)
#
# fig.add_layer(ivy.vis.layers.add_circles, mr_tree.leaves(),
#               colors = cols, size=4)

trueSlowQ = np.array([[-.1,.1], [.1,-.1]])
trueFastQ = np.array([[-.8,.8], [.8,-.8]])
trueQs = np.array([trueSlowQ, trueFastQ])

fastNodes = np.array([ n.ni for n in mr_tree.mrca(mr_tree.grep("f")).preiter()])
slowNodes = np.array([n.ni for n in mr_tree.descendants()if not n.ni in fastNodes])

trueLocs = [fastNodes, slowNodes]

l = discrete.mk_multi_regime(mr_tree, mr_chars, trueQs, trueLocs, pi="Equal")

m = bayesian_models.create_multi_mk_model(mr_tree, mr_chars, Qtype="ER", pi="Equal", nregime=2)



mc = pymc.MCMC(m)

mc.sample(5000, burn=200)


out = {"Qparams":mc.trace("Qparams_scaled")[:],
       "switch":mc.trace("switch")[:]}


switchids = [ int(i) for i in out["switch"] ]
switchnodes = [ mr_tree[i] for i in switchids ]
#
# with open("/home/cziegler/src/christie-master/ivy/ivy/tests/scripts/multiregime_test.pickle",
#           "wb") as handle:
#     pickle.dump(out, handle)


Q1 = [ i[0][0] for i in out["Qparams"] ] # fast
Q2 = [ i[0][1] for i in out["Qparams"] ] # slow


q1p = plt.plot(Q1)
plt.figure()
q2p = plt.plot(Q2)
plt.figure()
swp = plt.plot(out["switch"])

# Estimated Q values

np.percentile(Q1, 50)


np.percentile(Q2, 50)

# Estimated switchpoint
scipy.stats.mode(out["switch"])[0]


fig = treefig(mr_tree)
fig.toggle_branchlabels()
fig.tip_chars(mr_chars, store="tips")
fig.add_layer(layers.add_node_heatmap, switchnodes, store="switch")



# Comparing to an otherwise identical single-regime model
single_regime = bayesian_models.create_mk_model(mr_tree, mr_chars, Qtype="ER",
                                                pi="Equal")
single_regime_mc = pymc.MCMC(single_regime)
single_regime_mc.sample(1000)

Q = single_regime_mc.trace("Qparams_scaled")[:]

Qavg = np.percentile(Q,50)
# 0.38377105527703087
# In between the two rates, but closer to the slow rate
# (which takes up most of the tree) as expected



# Comparing the likelihoods of the MAP estimates,
# for lack of a better comparison
Q1avg = np.percentile(Q1, 50)
Q2avg = np.percentile(Q2, 50)

multiregime_Q = np.array([[[-Q1avg,Q1avg],
                           [Q1avg,-Q1avg]],
                           [[-Q2avg,Q2avg],
                             [Q2avg,-Q2avg]]])
multiregime_locs = discrete.locs_from_switchpoint(mr_tree, mr_tree[int(scipy.stats.mode(out["switch"])[0])])

multiregime_likelihood = discrete.mk_multi_regime(mr_tree, mr_chars,
                         multiregime_Q, multiregime_locs,
                         pi="Equal")

singleregime_Q = np.array([[-Qavg,Qavg],
                             [Qavg,-Qavg]])

singleregime_likelihood = discrete.mk(mr_tree, mr_chars, singleregime_Q,
                                      pi="Equal")


multiregime_likelihood
singleregime_likelihood


# Likelihood ratio test: we can do this because the 1-regime model
# is a special case of the 2-regime model

lr = -2*singleregime_likelihood + 2*multiregime_likelihood

# pchisq(19.4675329194382, 1, lower.tail=FALSE)
# 1.023242e-05
# Strong support for two regimes



# We want to test what happens if this model is given a single-regime
# tree and character set. Does it incorrectly give evidence for two regimes?


sr_chars = [1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1]






sr_m = bayesian_models.create_multi_mk_model(mr_tree, sr_chars, Qtype="ER", pi="Equal", nregime=2)






sr_mc = pymc.MCMC(sr_m)
sr_mc.sample(2000, burn=200)
sr_switchnodes = list(sr_mc.trace("switch")[:])
sr_switch = scipy.stats.mode(sr_mc.trace("switch")[:])[0]

sr_qs = sr_mc.trace("Qparams_scaled")[:]

sr_q1s = ([i[0][0] for i in sr_qs])
sr_q2s = ([i[0][1] for i in sr_qs])

sr_q1 = np.median(sr_q1s)
sr_q2 = np.median(sr_q2s)

sr_switchnodes = [mr_tree[int(i)] for i in sr_switchnodes]

sr_fig = treefig(mr_tree)
sr_fig.toggle_branchlabels()
sr_fig.tip_chars(sr_chars, store="tips", colors=["black", "red"])
sr_fig.add_layer(layers.add_node_heatmap, sr_switchnodes, store="switch")


# Now to fit a single-regime model

sr_sm = bayesian_models.create_mk_model(mr_tree, sr_chars, Qtype="ER", pi="Equal")
sr_smc = pymc.MCMC(sr_sm)
sr_smc.sample(2000, burn=200)

srsm_q = np.median(sr_smc.trace("Qparams_scaled")[:])

sr_multi_Qs = np.array([[[-sr_q1, sr_q1],[sr_q1, -sr_q1]],
                        [[-sr_q2, sr_q2],[sr_q2, -sr_q2]]])
sr_single_Q = np.array([[-srsm_q,srsm_q],[srsm_q,-srsm_q]])


sr_multi_locs = discrete.locs_from_switchpoint(mr_tree, mr_tree[int(sr_switch[0])])
sr_multi_like = discrete.mk_multi_regime(mr_tree, sr_chars, sr_multi_Qs, sr_multi_locs,
                                         pi="Equal")
sr_single_like = discrete.mk(mr_tree, sr_chars, sr_single_Q, pi="Equal")

lr = -2*singleregime_likelihood -sr_q1, sr_q1, sr_q1, -sr_q1 2*multiregime_likelihood


#
# # mk model 2
#
# fig = treefig(mr_tree)
# fig.toggle_branchlabels()
# fig.tip_chars(mr_chars, store="tips")
# fig.add_layer(layers.add_node_heatmap, switchnodes, store="switch")




m2 = create_multi_mk_model_2(mr_tree, mr_chars, Qtype="ER", pi="Equal")
m2mc = pymc.MCMC(m2)
m2mc.sample(100000)

m2mc.trace("nswitches")[:]



#


# tests

t_locs = discrete.locs_from_switchpoint(mr_tree, 262)



nshifts(mr_tree, get_indices(t_locs))
