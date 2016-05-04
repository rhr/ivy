import ivy
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy import cluster

from ivy.vis import layers
from ivy.sim.discrete_sim import sim_discrete
from ivy.chars import mk, hrm, recon

from ivy.interactive import *

tree = ivy.tree.read("support/hrm_600tips.newick")


# Generating model
Q = np.array([[-0.02,0.01,0.01,0],
              [0.01,-.02,0,0.01],
              [0.01,0,-.11,0.1],
              [0,0.01,0.1,-.11]])
# Creating the simulation
rseed = 18
simtree = ivy.sim.discrete_sim.sim_discrete(tree, Q, anc=0, rseed=rseed)
fig = treefig(simtree)
fig.add_layer(ivy.vis.layers.add_branchstates)

chars = [i.sim_char["sim_state"] for i in simtree.leaves()]
chars =  [i%2 for i in chars]


##########################
# 8-param MLE
##########################
hrm_mle = hrm.fit_hrm(tree, chars, nregime=2, Qtype="ARD")
Q = hrm_mle["Q"].copy()


#########################
# Clustering
#########################

alt_mods = cluster_models(tree, chars, Q, nregime)


###################
# Merging
###################
out = hrm.pairwise_merge(tree, chars, Q, 2)
