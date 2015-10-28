import ivy
import numpy as np
from ivy.chars import discretetraits

# Estimating pi from the stationary distribution of Q

# Method A: calculate pi every time the likelihood for a new Q matrix
# is calculated, where pi is taken from the stationary distribution of the
# Q matrix that is being used in the likelihood calculation, and then getting
# the likelihood using that value for pi.


# Method B: estimate the Q matrix under the assumption of a
# flat prior on the root and then take the stationary distribution of
# that Q matrix to use as pi. Then use that pi value in all further
# likelihood calculations and estimate Q again using the likelihood
# values generated from using that value for pi.


# Tree generated in R, history simulated in R using
# the following Q matrix:

trueQ = np.array([[-0.55, 0.4, 0.15],
                  [0.05, -0.30, 0.25],
                  [0.15, 0.1, -0.25]], dtype=np.double)


tree = ivy.tree.read("../support/randtree100tipsscale5.newick")

chars = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1,
0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0,
0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]

# Method A:
qA, lA = discretetraits.fitMkARD(tree, chars, pi="Equilibrium")

piA = discretetraits.qsd(qA) # The pi used in the final likelihood calculation
                             # is the stationary distribution of the
                             # fitted Q matrix

# Method B
qB, lB, piB = discretetraits.fitMk(tree, chars, Q="ARD", pi="Equilibrium")


# Method A seems to overestimate rates
# It has a more evenly distributed prior at the root
