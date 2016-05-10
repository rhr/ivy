import ivy
import numpy as np
import scipy
from ivy.chars import mk
from scipy.special import binom
from ivy.chars.expokit import cyexpokit
import math
import cProfile

tree = ivy.tree.read("support/randtree100tipsscale2.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]

Q = np.array([[-2.09613850e-01, 1.204029e-01, 8.921095e-02],
              [5.654382e-01, -5.65438217e-01, 1.713339e-08],
              [2.415020e-06, 5.958744e-07, -3.01089440e-06]])


out = mk.fit_Mk(tree, chars, Q="ARD", pi="Equal")

cProfile.run('mk.fit_Mk(tree, chars, Q="ARD", pi="Equal")')

Q = np.array([[-2.09613850e-01, 1.204029e-01, 8.921095e-02],
              [5.654382e-01, -5.65438217e-01, 1.713339e-08],
              [2.415020e-06, 5.958744e-07, -3.01089440e-06]])
calculatedLikelihood = mk.mk(tree, chars, Q,
                                         pi = "Fitzjohn")




tree = ivy.tree.read("support/randtree100tips.newick")

chars = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

out = mk.fit_Mk(tree, chars, Q="Sym", pi="Equilibrium")
