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
#
# mod_graph = pickle.load(open("mod_graph_223.p", "rb"))
#
# qs = make_qmat_stoch(mod_graph)
#
# mod_mc = pymc.MCMC([qs])
# mod_mc.use_step_method(QmatMetropolis, qs, graph=mod_graph)
#
# mod_mc.sample(1000)
#
# sample_from_prior = mod_mc.trace("qmat_stoch")[:]

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
qmat_MCMC = pymc.MCMC([qmat_stoch])
qmat_MCMC.use_step_method(QmatMetropolis, qmat_stoch,nparam, nchar, nregime)



qmat_MCMC.sample(1000000)

modtrace = qmat_MCMC.trace("qmat_stoch")[:]

#############################
# Model selection with 3 regimes
##############################
nregime = 3
nparam = 3
modspace_mc_3r = hrm_allmodels_bayes(tree, chars, nregime, nparam)
modspace_mc_3r.sample(20000, burn=2000, thin=2)

modspace_mc_3r_slow = modspace_mc_3r.trace("slow")[:]
modspace_mc_3r_alpha = modspace_mc_3r.trace("paramscale_0")[:]
modspace_mc_3r_beta = modspace_mc_3r.trace("paramscale_1")[:]


plt.hist(modspace_mc_3r_slow)
plt.figure()
plt.hist(modspace_mc_3r_alpha)
plt.figure()
plt.hist(modspace_mc_3r_beta)
modspace_mc_3r_modcount = collections.Counter([tuple(i) for i in modspace_mc_3r.trace("mod")[:]])

#############################
# Modelselection with 3 regimes as generating model
#############################
generatingQ = np.array([[-.02,0,0.01,0,0.01,0],
                        [0,-0.02,0,0.01,0,0.01],
                        [0.01,0,-0.03,0.01,0.01,0],
                        [0,0.01,0.01,-0.03,0,0.01],
                        [0.01,0,0.01,0,-0.12,0.1],
                        [0,0.01,0,0.01,0.1,-0.12]])
simtree = ivy.sim.discrete_sim.sim_discrete(tree, generatingQ, anc=0)
chars = [i.sim_char["sim_state"] for i in simtree.leaves()]
chars =  [i%2 for i in chars]

regime = 3
nparam = 3
modfit_3r = hrm_allmodels_bayes(tree, chars, nregime, nparam)
modfit_3r.sample(20000, burn=2000, thin=2)


modfit_3r_modcount = collections.Counter([tuple(i) for i in modfit_3r.trace("mod")[:]])
modfit_3r_slow = modfit_3r.trace("slow")[:]
modfit_3r_alpha = modfit_3r.trace("paramscale_0")[:]
modfit_3r_beta = modfit_3r.trace("paramscale_1")[:]


plt.hist(modfit_3r_slow)
plt.figure()
plt.hist(modfit_3r_alpha)
plt.figure()
plt.hist(modfit_3r_beta)
############################
# Model selection: sample from prior
############################
nchar = 6
nregime = 3
nparam = 3
nobschar = 2
minp = pscore(tree, chars)
treelen = sum([n.length for n in tree.descendants()])
slow = pymc.Exponential("slow", beta=treelen/minp)

#Parameters:
paramscales = [None]*(nparam-1)
for p in range(nparam-1):
    paramscales[p] =  ShiftedGamma(name = "paramscale_{}".format(str(p)), shape = GAMMASHAPE, shift=1)

mod = make_qmat_stoch(nobschar = nobschar,nregime=nregime,
                      nparam=nparam, mod_order_list = list(itertools.product(range(nparam+1), repeat = nobschar**2-nobschar)),
                      name="mod")
mod_nodata = pymc.MCMC([slow, paramscales, mod])

mod_nodata.use_step_method(QmatMetropolis, mod, nparam, nchar, nregime)

mod_nodata.sample(20000, burn=2000, thin=2)

nodata_modcount = collections.Counter([tuple(i) for i in mod_nodata.trace("mod")[:]])
nodata_slow = mod_nodata.trace("slow")[:]
nodata_alpha = mod_nodata.trace("paramscale_0")[:]
nodata_beta = mod_nodata.trace("paramscale_1")[:]

plt.figure()
plt.hist(nodata_slow, color="red")
plt.figure()
plt.hist(nodata_alpha, color="red")
plt.figure()
plt.hist(nodata_beta, color="red")



###############################################
# Well-behaved model testing
###############################################
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
from ivy.interactive import *

tree = ivy.tree.read("support/hrm_600tips.newick")

# Two-regime, two parameter
Q = np.array([[-0.02,0.01,0.01,0],
              [0.01,-.02,0,0.01],
              [0.01,0,-.11,0.1],
              [0,0.01,0.1,-.11]])
rseed = 18
simtree = ivy.sim.discrete_sim.sim_discrete(tree, Q, anc=0, rseed=rseed)
fig = treefig(simtree)
fig.add_layer(ivy.vis.layers.add_branchstates)

chars = [i.sim_char["sim_state"] for i in simtree.leaves()]
chars =  [i%2 for i in chars]

nchar = 4
nregime = 2
nparam = 2
nobschar = 2

####
# Without data
###

minp = pscore(tree, chars)
treelen = sum([n.length for n in tree.descendants()])
slow = pymc.Exponential("slow", beta=treelen/minp)

#Parameters:
paramscales = [None]*(nparam-1)
for p in range(nparam-1):
    paramscales[p] =  ShiftedGamma(name = "paramscale_{}".format(str(p)), shape = GAMMASHAPE, shift=1)

mod = make_qmat_stoch(nobschar = nobschar,nregime=nregime,
                      nparam=nparam, mod_order_list = list(itertools.product(range(nparam+1), repeat = nobschar**2-nobschar)),
                      name="mod")
mod_nodata_2r = pymc.MCMC([slow, paramscales, mod])

mod_nodata_2r.use_step_method(QmatMetropolis, mod, nparam, nchar, nregime)

mod_nodata_2r.sample(20000, burn=2000, thin=2)

mod_nodata_2r_modcount = collections.Counter([tuple(i) for i in mod_nodata_2r.trace("mod")[:]])
mod_nodata_2r_slow = mod_nodata_2r.trace("slow")[:]
mod_nodata_2r_alpha = mod_nodata_2r.trace("paramscale_0")[:]

plt.figure()
plt.hist(mod_nodata_2r_slow, color="red")
plt.figure()
plt.hist(mod_nodata_2r_alpha, color="red")


#####
# MLE to pick seed value
#####
mod_2r_MLE = hrm.fit_hrm_mkARD(tree, chars, 2, pi="Equal", constraints="Rate", orderedRegimes=False)
MLEQ = np.array([[-0.12103,  0.10938,  0.01165,  0.     ],
                 [ 0.07892, -0.07892,  0.     ,  0.     ],
                 [ 0.03479,  0.     , -0.03479,  0.     ],
                 [ 0.     ,  0.     ,  0.69436, -0.69436]])


recon = ivy.chars.recon.recon_discrete(tree, chars, Q=MLEQ, nregime = 2)

modelseed = (1,1,2,2,1,1,1,1)

modfit_2r = hrm_allmodels_bayes(tree, chars, nregime, nparam, modseed=modelseed, dbname="modfit_2r.p")
modfit_2r.sample(1000000, burn=100000, thin=10)

modfit_2r_modcount = collections.Counter([tuple(i) for i in modfit_2r.trace("mod")[:]])
modfit_2r_slow = modfit_2r.trace("slow")[:]
modfit_2r_alpha = modfit_2r.trace("paramscale_0")[:]


modfit_2r_modcount.most_common()[:10]



####################################
# Exhaustive MLE model selection
####################################
modMLE_2r_brvar = hrm.fit_hrm_distinct_regimes(tree, chars, 2, 2, br_variable=True,
            out_file = "modMLE_2r_brvar")
modMLE_2r_brcon = fit_hrm_distinct_regimes(tree, chars, 2, 2, br_variable=False,
            out_file = "modMLE_2r_brcon")

mods_sorted = collections.OrderedDict(sorted(modMLE_2r.items(), key = lambda x: x[1][0]))
mods_sorted_brcon = collections.OrderedDict(sorted(modMLE_2r_brcon.items(), key = lambda x: x[1][0]))
