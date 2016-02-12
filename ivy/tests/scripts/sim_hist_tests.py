# Is the phytools method of simulating a character on a tree
# overestimating the number of changes?
import ivy
from ivy.chars import mk
import numpy as np
import math
import scipy
import scipy.stats
import random
from ivy.sim import discrete_sim

tree = ivy.tree.read("../support/randtree100tips.newick")
Q = np.array([[-0.8,0.8],[0.2,-0.2]])

NREP = 1000

sim_phy = [None]*NREP
for i in range(NREP):
    sim_phy[i] = discrete_sim.sim_discrete_phytools(tree,Q)

sim_niels = [None]*NREP
for i in range(NREP):
    sim_niels[i] = discrete_sim.sim_discrete_nielsen(tree,Q)


# Count number of state changes
sim_phy_changes = [0]*NREP
for i in range(NREP):
    for p in sim_phy[i].descendants():
        if p.sim_char["sim_state"] != p.parent.sim_char["sim_state"]:
            sim_phy_changes[i] += 1


# Count number of state changes
sim_niels_changes = [0]*NREP
for i in range(NREP):
    for p in sim_niels[i].descendants():
        if p.sim_char["sim_state"] != p.parent.sim_char["sim_state"]:
            sim_niels_changes[i] += 1

# Expected # of changes
sum([n.length for n in tree.descendants()])

np.percentile(sim_phy_changes, [2.5,50,97.5 ])
plt.hist(sim_phy_changes)
np.percentile(sim_niels_changes, [2.5,50,97.5 ])
plt.hist(sim_niels_changes)
