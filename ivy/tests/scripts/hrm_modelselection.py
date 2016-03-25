import ivy
import numpy as np
import math
import itertools
import pymc
import random
import collections
import networkx as nx
import scipy
import pickle
from ivy.chars.hrm_bayesian_graph import *


# unique_mods = unique_models(2,2,3)
# mod_graph = make_model_graph(unique_mods)
#
# pickle.dump(mod_graph, open("mod_graph_223.p", "wb"))

mod_graph = pickle.load(open("mod_graph_223.p", "rb"))

qs = make_qmat_stoch(mod_graph)

mod_mc = pymc.MCMC([qs])
mod_mc.use_step_method(QmatMetropolis, qs, graph=mod_graph)

mod_mc.sample(1000)

sample_from_prior = mod_mc.trace("qmat_stoch")[:]


pos = nx.random_layout(mod_graph)
fig, ax = plt.subplots()
segs = []
for node in mod_graph.node.keys():
    for cnode in mod_graph[node].keys():
        segs.append([pos[node], pos[cnode]])
ax.add_collection(matplotlib.collections.LineCollection(segs, colors=(0,0,0,0.005)))

# coloring counts
tracesegs = []
for i,node in enumerate(sample_from_prior[:-1]):
    tracesegs.append([pos[tuple(node)], pos[tuple(sample_from_prior[i+1])]])
ax.add_collection(matplotlib.collections.LineCollection(tracesegs, colors=(0,0,1,0.005)))

def trace_model_graph(trace, graph):
    """
    Plot trace on graph of models
    """
    pos = nx.random_layout(graph)
    fig, ax = plt.subplots()
    segs = []
    for node in graph.node.keys():
        for cnode in graph[node].keys():
            segs.append([pos[node], pos[cnode]])
    ax.add_collection(matplotlib.collections.LineCollection(segs, colors=(0,0,0,0.005)))

    # coloring counts
    tracesegs = []
    for i,node in enumerate(trace[:-1]):
        tracesegs.append([pos[tuple(node)], pos[tuple(trace[i+1])]])
    ax.add_collection(matplotlib.collections.LineCollection(tracesegs, colors=(0,0,1,0.05)))




count = collections.Counter([tuple(i) for i in mod_mc.trace("qmat_stoch")[:]])

count.most_common()[:15]
sum([i[1] for i in count.most_common()[:15]])


#################################################
tree = ivy.tree.read("support/hrm_600tips.newick")
chars = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
                1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]
nregime = 2
nparam = 2
pi = "Equal"






def Lognorm(mu, sigma, name="Lognorm"):
    @pymc.stochastic(name = name)
    def Lognorm_mu_sigma(value=1, mu=mu, sigma=sigma):
        """
        Lognormal paramaterized with mu and sigma
        """
        p = 1/(value*sigma*np.sqrt(2*np.pi)) * np.exp(-1*((np.log(value)-mu)**2/(2*sigma**2)))
        return np.log(p)
    return Lognorm_mu_sigma


mu, sigma = lognormal_percentile(v1, v2, p)
alpha = Lognorm(mu, sigma, "alpha")

mc = pymc.MCMC([alpha])
mc.sample(1000)
plt.hist(mc.trace("alpha")[:])


gamma_shape = 5.43
# Gamma prior
alpha = pymc.Gamma("alpha", gamma_shape, 1)

def ShiftedGamma(shape, shift = 1, name="ShiftedGamma"):
    @pymc.stochastic(name=name)
    def shifted_gamma(value=2, shape=shape):
        return pymc.gamma_like(value-shift, shape, 1)
    return shifted_gamma
gamma_shape = 3.55
alpha = ShiftedGamma(shape=gamma_shape, name="alpha")

mc = pymc.MCMC([alpha])
mc.sample(1000)
hist(mc.trace("alpha")[:])



modspace_mc2 = hrm_allmodels_bayes(tree, chars, nregime, nparam, mod_graph=mod_graph)
modspace_mc2.sample(20000, burn=2000, thin=2)

trace_model_graph(modspace_mc2.trace("mod")[:],mod_graph)

count = collections.Counter([tuple(i) for i in modspace_mc2.trace("mod")[:]])


strace = modspace_mc.trace("slow")[:]
mtrace = [strace[i] * modspace_mc.trace("alpha")[i] for i in range(len(strace))]
ftrace = [mtrace[i] * modspace_mc.trace("beta")[i] for i in range(len(strace))]

alphatrace = modspace_mc.trace("alpha")[:]
betatrace = modspace_mc.trace("beta")[:]

modtrace = modspace_mc.trace("mod")[:]


# Building model graph on the fly
modspace_mc2 = hrm_allmodels_bayes(tree, chars, nregime, nparam, mod_graph=mod_graph)
modspace_mc2.sample(20000, burn=2000, thin=2)
