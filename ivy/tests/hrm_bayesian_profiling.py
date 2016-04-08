import ivy
import numpy as np
import math
import itertools
import pymc
import random
import collections
import scipy

import matplotlib
import matplotlib.pylab as plt
import cProfile
from ivy.chars.hrm_bayesian import hrm_allmodels_bayes
from ivy.chars.expokit import cyexpokit
# Setting up simple model to test

tree = ivy.tree.read("support/hrm_600tips.newick")
Q = np.array([[-0.02,0.01,0.01,0],
              [0.01,-.02,0,0.01],
              [0.01,0,-.11,0.1],
              [0,0.01,0.1,-.11]])
random.seed(2)
simtree = ivy.sim.discrete_sim.sim_discrete(tree, Q, anc=0)
# fig = treefig(simtree)
# fig.add_layer(ivy.vis.layers.add_branchstates)

chars = [i.sim_char["sim_state"] for i in simtree.leaves()]
chars =  [i%2 for i in chars]

modseed = (0,0,2,2,1,1,1,1)

mcmc = hrm_allmodels_bayes(tree, chars, 2, 3, modseed)

cProfile.run("mcmc.sample(100)")


######################
# Profiling cython code
#####################
import ivy
import numpy as np
import math
import random

import matplotlib
import matplotlib.pylab as plt
import cProfile
from ivy.chars.expokit import cyexpokit
import line_profiler
from Cython.Compiler.Options import directive_defaults

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

tree = ivy.tree.read("support/hrm_600tips.newick")
Q = np.array([[-0.02,0.01,0.01,0],
              [0.01,-.02,0,0.01],
              [0.01,0,-.11,0.1],
              [0,0.01,0.1,-.11]])
random.seed(2)
simtree = ivy.sim.discrete_sim.sim_discrete(tree, Q, anc=0)
# fig = treefig(simtree)
# fig.add_layer(ivy.vis.layers.add_branchstates)

chars = [i.sim_char["sim_state"] for i in simtree.leaves()]
chars =  [i%2 for i in chars]

ar = ivy.chars.hrm.create_hrm_ar(tree, chars, 2)
cyexpokit.dexpm_tree_preallocated_p_log(Q, ar["t"], ar["p"])
profile = line_profiler.LineProfiler(cyexpokit.cy_mk_log)

profile.runcall(cyexpokit.cy_mk_log, ar["nodelist"],ar["p"],4)

profile.print_stats()
