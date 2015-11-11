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



tree = ivy.tree.read("../tests/support/randtree100tipsscale2.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

modelout = ivy.chars.bayesian_models.fit_mk_bayes(tree, chars, Qtype="ARD", pi="Equal")
