import ivy
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle

from ivy.interactive import *
from ivy.vis import layers
from ivy.sim import discrete_sim
from colour import Color
from matplotlib.colors import LinearSegmentedColormap

fast = 1.0
slow = 0.01
br = 0.02

Q = ivy.chars.hrm.fill_Q_matrix(2,2,[slow,fast],[br], Qtype="Simple")

tree = ivy.tree.read("support/hrm_300tips.newick")
#random.seed(1)
#simtree = discrete_sim.sim_discrete(tree, Q, anc=0)
simtree = pickle.load(open("sim_tree.p","rb"))
chars_unhidden = [ n.sim_char["sim_state"] for n in simtree.leaves()]
chars = [ n % 2 for n in chars_unhidden]
nchar = 4
trueSimFig = treefig(simtree)
trueSimFig.add_layer(layers.add_branchstates)


hrm_MLE = ivy.chars.hrm.fit_hrm_mkSimple(tree, chars, 2, pi="Equal")


#t = ivy.chars.anc_recon.anc_recon(tree, chars, np.array([[-0.5,0.5],[0.5,-0.5]]), pi="Equal")
#pickle.dump(t, open("prev_ancrecon_results.p", "wb"))

t = pickle.load(open("prev_ancrecon_results.p", "rb"))
recon = ivy.chars.anc_recon.anc_recon_discrete(tree, chars,hrm_MLE[0], pi="Equal", nregime = 2)


fig = treefig(tree)

cols = ["pink", "blue", "red", "darkblue"]

fig.add_layer(layers.add_ancestor_noderecon, recon, colors = cols)


########################
# Visualizing likelihood
########################

def twoS_twoR_colormaker(lik):
    """
    Given node likelihood, return appropriate color

    State 0 corresponds to red, state 1 corresponds to blue
    Regime 1 corresponds to grey, regime 2 corresponds to highly saturated
    """
    s0 = sum([lik[0], lik[2]])
    s1 = sum([lik[1], lik[3]])

    r0 = sum([lik[0], lik[1]])
    r1 = sum([lik[2], lik[3]])

    sat = r1 + (1-r1)*0.3

    col = Color(rgb=(s0,0,s1))
    col.saturation = sat
    return col



cols = [twoS_twoR_colormaker(i[:nchar]).rgb for i in recon]

fig = treefig(tree)
fig.add_layer(layers.add_circles, list(tree.preiter()),colors=cols)


#################
# Gradient bars
#################
c1 = (0,0,1)
c2 = (1,0,0)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.axis([0,1,-50,200])

p1 = (0.1,0)
p2 = (0.9,0)

cust_cm = LinearSegmentedColormap.from_list("cust_cm",[c1, c2])
width = 10

nsegs = 255
seglen = abs(p1[0] - p2[0])/nsegs

pos = zip(np.arange(p1[0], p2[0], seglen), [p1[1]]*nsegs)
pos.append(p2)

segs = [[pos[i],pos[i+1]] for i in range(nsegs)]

vals = np.arange(0,1,seglen)
lc = LineCollection(segs, cmap=cust_cm, linewidths=width)
lc.set_array(vals)
ax.add_collection(lc)

c1 = twoS_twoR_colormaker([1,0,0,0])
c2 = twoS_twoR_colormaker([0,1,0,0])
c3 = twoS_twoR_colormaker([0,0,1,0])
c4 = twoS_twoR_colormaker([0,0,0,1])


fig,axes = plt.subplots()
