# Goal: find the MAP estimate for Q for a one-rate, two-state mk model using emcee
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


# slow_tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/Mk_slow_regime_tree.newick")
# fast_tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/Mk_fast_regime_tree.newick")
# fast_tree.length = 0.0001
# #fast_tree.children[0].length -= .0001
# #fast_tree.children[1].length -=.0001
#
# mr_tree = slow_tree.copy()
#
# x = (0.5001 - slow_tree["s99"].rootpath_length(slow_tree["292"]))/slow_tree["292"].length
#
#
# switch_node = mr_tree["292"].bisect_branch(x)
#
#
# switch_node.add_child(fast_tree)
#
# mr_tree["f99"].rootpath_length()
#
#
# # multi-regime tree
mr_tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/Mk_two_regime_tree.newick")




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

with open("/home/cziegler/src/christie-master/ivy/ivy/tests/scripts/multiregime_test.pickle",
          "wb") as handle:
    pickle.dump(out, handle)


Q1 = [ i[0][0] for i in out["Qparams"] ] # fast
Q2 = [ i[0][1] for i in out["Qparams"] ] # slow

# Estimated Q values

np.percentile(Q1, 50)


np.percentile(Q2, 50)

# Estimated switchpoint
scipy.stats.mode(out["switch"])[0]

# Comparing to an otherwise identical single-regime model
single_regime = create_mk_model(mr_tree, mr_chars, Qtype="ER",
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


print multiregime_likelihood
print singleregime_likelihood


# Likelihood ratio test: we can do this because the 1-regime model
# is a special case of the 2-regime model

lr = -2*singleregime_likelihood + 2*multiregime_likelihood









# We want to test what happens if this model is given a single-regime
# tree and character set. Does it incorrectly give evidence for two regimes?
