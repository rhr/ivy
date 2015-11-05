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
from matplotlib import pyplot as plt

import emcee
import corner

tree = ivy.tree.read("support/randtree100tipsscale2.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

t = np.array([node.length for node in tree.postiter() if not node.isroot], dtype=np.double)

p = np.empty([len(t),2,2], dtype = np.double, order="C")


def log_prior(Qparam):
    lam = 1.0
    if Qparam[0] < 0 or Qparam[0]>200:
        return -np.inf  # log(0)
    else:
        return lam*math.exp(-lam*Qparam[0]) # Exponential prior

log_likelihood = discrete.create_likelihood_function_mk(tree, chars, Qtype="ER", pi="Equal", min=False)

def log_posterior(Qparam):
    return log_prior(Qparam)+log_likelihood(Qparam)


ndim = 1
nwalkers = 50
nburn = 1000
nsteps = 2000

# starting_guesses = np.random.random((nwalkers, ndim))
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
# %time sampler.run_mcmc(starting_guesses, nsteps)
# print("done")
#
# # Looking at the results
# samples = sampler.chain[:,nburn:,:].reshape((-1,ndim))
#
# fig = corner.corner(samples)
# fig.show()
#
# np.percentile(samples, 50)
#
#
#
#
#
# # Two-rate, two-state matrix
# def log_prior(Qparams):
#     if any(Qparams < 0) or any(Qparams > 200):
#         return -np.inf  # log(0)
#     else:
#         return 0.0 # Flat prior
#
# log_likelihood = discrete.create_likelihood_function_mk(tree, chars, Qtype="ARD", pi="Equal", min=False)
#
# def log_posterior(Qparam):
#     return log_prior(Qparam)+log_likelihood(Qparam)
#
#
# ndim = 2
# nwalkers = 50
# nburn = 1000
# nsteps = 2000
#
# starting_guesses = np.random.random((nwalkers, ndim))
#
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior)
# %time sampler.run_mcmc(starting_guesses, nsteps)
# print("done")
#
# # Looking at the results
#
# fig,ax = plt.subplots(1,1)
# res = ax.plot(sampler.chain[:,:,0].T, '-', color='k', alpha=0.3)
#
# samples = sampler.chain[:,nburn:,:].reshape((-1,ndim))
#
# fig = corner.corner(samples)
# fig.show()
# q1, q2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                              zip(*np.percentile(samples, [16, 50, 84],
#                                                 axis=0)))



# One-rate, two-state, two-regime model: Fixed regime

r1 = [ i for i,n in enumerate(tree.postiter()) if (not n in list(tree[15].postiter())) and (not n.isroot)]
r2 = [ i for i in range(198) if not i in r1]

locs = [r1, r2]


# Visualizing the tree:
from ivy.interactive import *
fig = treefig(tree)
fig.add_layer(ivy.vis.layers.add_circles, tree.leaves(),
              colors=[ ["black","red"][s] for s in chars ], size=5)

def log_prior(Qparams):
    if any(Qparams < 0) or any(Qparams > 200):
        return -np.inf  # log(0)
    else:
        return 0.0 # Flat prior

log_likelihood = discrete.create_likelihood_function_multimk(tree, chars, Qtype="ER",
                  locs=locs, pi="Equal", min=False)


def log_posterior(Qparams):
    return log_prior(Qparams) + log_likelihood(Qparams)

ndim = 2
nwalkers = 50
nburn = 1000
nsteps = 2000

starting_guesses = np.random.random((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads = 5)
%time sampler.run_mcmc(starting_guesses, nsteps)
print("done")

# Looking at the results

fig,ax = plt.subplots(1,1)
res = ax.plot(sampler.chain[:,:,0].T, '-', color='k', alpha=0.3)

samples = sampler.chain[:,nburn:,:].reshape((-1,ndim))

fig = corner.corner(samples)
fig.show()
q1, q2 = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                             zip(*np.percentile(samples, [16, 50, 84],
                                                axis=0)))
