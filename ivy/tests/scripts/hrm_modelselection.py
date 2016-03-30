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
from ivy.chars.hrm_bayesian import *


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


gamma_shape = 3.55


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

modspace_mc2 = hrm_allmodels_bayes(tree, chars, nregime, nparam, mod_graph=mod_graph)
modspace_mc2.sample(12000, burn=2000, thin=10)




## Simple Mk bayesian model selection
tree = ivy.tree.read("support/hrm_600tips.newick")
simtree = ivy.sim.discrete_sim.sim_discrete(tree, np.array([[-0.01,0.01],[0.1,-0.1]]))
fig = treefig(simtree)
fig.add_layer(ivy.vis.layers.add_branchstates)

chars = [t.sim_char["sim_state"] for t in simtree.leaves()]
mk_mod = mk_allmodels_bayes(tree, chars, nparam=2, pi="Equal")


mk_mod.sample(5000, burn=1000)
s = mk_mod.trace("slow")[:]
alpha = mk_mod.trace("paramscale_0")[:]
f = [s[i]*alpha[i] for i in range(len(s))]

print np.percentile(s, [2.5,50,97.5])
print np.percentile(alpha, [2.5,50,97.5])
print np.percentile(f, [2.5,50,97.5])


modcount = collections.Counter([tuple(i) for i in mk_mod.trace("mod")[:]])




mk_mod_gammaprior = mk_allmodels_bayes(tree, chars, nparam=2, pi="Equal")

mk_mod_gammaprior.sample(5000, burn=1000)
s_gammaprior = mk_mod_gammaprior.trace("slow")[:]
alpha_gammaprior = mk_mod_gammaprior.trace("paramscale_0")[:]
f_gammaprior = [s[i]*alpha[i] for i in range(len(s))]

print np.percentile(s_gammaprior, [2.5,50,97.5])
print np.percentile(alpha_gammaprior, [2.5,50,97.5])
print np.percentile(f_gammaprior, [2.5,50,97.5])


modcount = collections.Counter([tuple(i) for i in mk_mod_gammaprior.trace("mod")[:]])


plt.hist(alpha_gammaprior)
plt.plot(x,(gamma.pdf(x,3.55, loc = 1)*4000))




###############################
# Model sampling
###############################
nobschar = 2
nparam = 3
nregime = 2
nchar = 4

n_wr = nobschar**2-nobschar
n_br = (nregime**2 - nregime)*nobschar

ncell = n_wr*nregime + n_br

qmat_stoch = make_qmat_stoch(ncells)
qmat_MCMC = pymc.MCMC(qmat_stoch)
qmat_MCMC.use_step_method(QmatMetropolis, qmat_stoch,nparam, nchar, nregime)
