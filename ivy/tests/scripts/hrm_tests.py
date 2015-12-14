import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt


tree = ivy.tree.read("/home/cziegler/src/christie-master/ivy/ivy/tests/support/hrm_300tips.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0]

nregime = 2
out = bayesian_models.hrm_bayesian(tree, chars, Qtype="Simple", nregime=2)

outMC = pymc.MCMC(out)
outMC.sample(10000, 400, 3)

plt.plot(outMC.trace("parA")[:])
plt.plot(outMC.trace("parB")[:])
plt.plot(outMC.trace("parC")[:])

for i in ["parA", "parB", "parC"]:
    print np.percentile(outMC.trace(i)[:], [2.5, 50, 97.5])
# True values are 0.01, 0.1, and 0.6









#
#
# # Campanulid dataset: analyzing with Multi-Regime Mk
# camp_tree = ivy.tree.read("/home/cziegler/Campanulid/Camp.BEST.d8.tre", format="newick")
#
# camp_data = ivy.tree.load_chars("/home/cziegler/Campanulid/Camp.GF2")
#
# camp_chars = [int(camp_data[n.label]["FORM"]) for n in camp_tree.leaves()]
# #
# # camp_fig.tip_chars(chars = [int(camp_data[n.label]["FORM"]) for n in camp_tree.leaves()],
# #                   colors = ["brown","green"])
#
#
# # First fit a basic Mk model
# camp_mk = discrete.fitMk(camp_tree, camp_chars, Q="ARD", pi="Fitzjohn")
