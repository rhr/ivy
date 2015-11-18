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
mr_tree = ivy.tree.read("support/Mk_two_regime_tree.newick")
