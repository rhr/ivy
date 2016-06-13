#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals

import math
import itertools
import random
import collections

import matplotlib
import matplotlib.pylab as plt
import numpy as np
import pymc
import scipy

from ivy.chars.recon import pscore
from ivy.chars import hrm, mk

"""
Functions for testing an hrm model by assigning parameters to different
cells of the Q matrix. Use custom step methods in pymc to traverse all
possible models.
"""

GAMMASHAPE = 3.55 # The shape parameter of a gamma distribution that is shifted by 1
                  # such that 95% of the distribution falls between 2 and 10
                  # with a scaling parameter of 1.


def unique_models(nchar, nregime, nparam):
    """
    Create a list of all possible models for a Q matrix with nchar distinct
    character states and nregime distinct regimes, filling all cells
    with nparam + 0.

    Args:
        nchar (int): Number of observed characters
        nregime (int): Number of regimes in model
        nparam (int): Number of unique parameters in model (not including 0)

    Returns:
        list: List of all unique models. Each item contains a tuple of the
          within-regime parameters and a tuple of the between-regime parameters.
    """
    param_ranks = list(range(nparam+1))
    n_wr = nchar**2 - nchar

    # Within-regime parameters
    distinct_regimes = list(itertools.permutations(param_ranks,n_wr))
    for p in param_ranks:
        distinct_regimes.append((p,p))
    wr = list(itertools.combinations(distinct_regimes,nregime))
    wr_flat = [ tuple([i for s in  x for i in s]) for x in wr ]

    # Between-regime parameters
    n_br = (nregime**2 - nregime)*nchar
    # Disallow the fastest parameter in between-regime transitions
    br = list(itertools.product(param_ranks[:-1], repeat = n_br))
    br.pop(0) # Need to handle "no transition" models separately

    mods = list(itertools.product(wr_flat, br))

    # "no transition" models
    nt_wr = [i+tuple([0]*n_wr) for i in distinct_regimes if set(i) != {0} ]
    nt_mods = [ (i, tuple([0]*n_br)) for i in nt_wr ]
    mods.extend(nt_mods)

    mods_flat = [ tuple([i for s in  x for i in s]) for x in mods ]

    return mods_flat


def new_hrm_model(mod, nparam, nchar, nregime, mod_order):
    """
    Return new, valid model by changing one parameter of current model
    """
    while 1:
        i = random.choice(list(range(len(mod))))
        v = random.choice([n for n in range(nparam+1) if not n==mod[i]])

        newmod = mod[:i] + (v,) + mod[i+1:]
        if is_valid_model(newmod, nparam, nchar, nregime, mod_order):
            return newmod


def is_valid_model(mod, nparam, nchar, nregime, mod_order):
    """
    Check if a given model is valid
    """
    nobschar = nchar//nregime
    n_wr = nobschar**2-nobschar
    n_br = (nregime**2-nregime)*nobschar
    all_params_zero = list(range(nparam+1))
    all_params = all_params_zero[1:]
    # Test if all params are present. If not, return false
    if not (list(set(mod)) == all_params_zero or list(set(mod)) == all_params):
        return False
    # Test if all regimes are connected
    if number_connected(mod, nchar, nregime, n_wr, n_br) < nregime-1:
        return False
    # Check order and identity of sub-matrices
    mod_indices = [mod_order[mod[i*n_wr:i*n_wr+n_wr]] for i in range(nregime)]
    if sorted(list(set(mod_indices))) != mod_indices:
        return False
    # Check that there are no fast parameters in BR transitions
    if nparam in mod[-n_br:]:
        return False
    return True


def number_connected(mod, nchar, nregime, n_wr, n_br):
    """
    Test how many regimes are connected in some way
    """
    br_mod = mod[int(n_wr*nregime):]
    n_pairs = int((nregime**2-nregime)/2)
    connections = [False]*n_pairs
    nobschar = nchar//nregime

    for c in range(n_pairs):
        connections[c] = set(br_mod[c*nobschar*2:c*nobschar*2+nobschar*2]) != {0}
    return sum(connections)


def make_qmat_stoch(nobschar,nregime,nparam,mod_order_list,modseed,name="qmat_stoch"):
    """
    Make a stochastic to use for a model-changing step
    """
    startingval = modseed

    @pymc.stochastic(dtype=tuple, name=name)
    def qmat_stoch(value = startingval):
        # Flat prior on model likelihood
        return 0
    return qmat_stoch


class QmatMetropolis(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new model
    """
    def __init__(self, stochastic, nparam, nchar, nregime):
        pymc.Metropolis.__init__(self, stochastic, scale=1.)
        self.nparam = nparam
        self.nchar = nchar
        self.nregime = nregime
        nobschar = nchar//nregime
        order = itertools.product(list(range(nparam+1)), repeat = nobschar**2-nobschar)
        self.mod_order = {m:i for i,m in enumerate(order)}
    def propose(self):
        cur_mod = self.stochastic.value
        new = new_hrm_model(cur_mod, self.nparam, self.nchar, self.nregime, self.mod_order)
        self.stochastic.value = new

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value


def ShiftedGamma(shape, shift = 1, name="ShiftedGamma"):
    """
    Gamma distribution that has been shifted by some constant.
    """
    @pymc.stochastic(name=name)
    def shifted_gamma(value=2, shape=shape):
        return pymc.gamma_like(value-shift, shape, 1)
    return shifted_gamma


def fill_model_Q(mod, Qparams, Q):
    """
    Create Q matrix given model and parameters.

    Args:
        mod (list): List of tuples. Code for the mode used. The first nregime
          tuples correspond to the within-regime transition rates for each
          regime. The last tuple corresponds to the between-regime transition
          rates. Ex. [(1, 2), (3, 4), (5, 6, 7, 8)] corresponds to a
          matrix of:
              [-,1,5,-]
              [2,-,-,6]
              [7,-,-,3]
              [-,8,4,-]
        Qparams (list): List of floats corresponding to the values for
          slow, medium, fast, etc. rates. The first is always 0
        Q (np.array): Pre-allocated Q matrix
    """
    nregime = len(mod)-1
    nobschar = Q.shape[0]//nregime
    nchar = Q.shape[0]
    Q.fill(0.0)
    for i in range(nregime):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        subQvals = [Qparams[x] for s in [(0,)+mod[i][k:k+nobschar] for k in range(0,len(mod[i])+1,nobschar)] for x in s]
        np.copyto(Q[subQ, subQ], [subQvals[x:x+nobschar] for x in range(0, len(subQvals), nobschar)])

    combs = list(itertools.combinations(list(range(nregime)),2))
    revcombs = [tuple(reversed(i)) for i in combs]
    submatrix_indices = [x for s in [[combs[i]] + [revcombs[i]] for i in range(len(combs))] for x in s]
    for i,submatrix_index in enumerate(submatrix_indices):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        nregimeswitch =(nregime**2 - nregime)*2
        np.fill_diagonal(Q[my_slice0,my_slice1],[Qparams[p] for p in mod[nregime][i*nobschar:i*nobschar+nobschar]])
    np.fill_diagonal(Q, -np.sum(Q,1))


def hrm_allmodels_bayes(tree, chars, nregime, nparam,modseed, pi="Equal",
                        dbname = None):
    """
    Use an MCMC chain to fit a hrm model with a limited number of parameters.

    The chain will step through different models, placing a prior on simpler
    models

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        nregime (int): Number of regimes
        nparam (int): Number of unique parameters to allow in a model
        modseed (tuple): Starting model for the MCMC chain. A tuple of ints.
           Must be a valid model or adjacent to at least one valid model.
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    assert nparam > 1, "nparam must be at least two"

    nobschar = len(set(chars))
    nchar = nobschar * nregime
    n_wr = nobschar**2-nobschar
    n_br = (nregime**2-nregime)*nobschar


    minp = pscore(tree, chars)
    treelen = sum([n.length for n in tree.descendants()])
    # Prior on slowest distribution (beta = 1/mean)
    slow = pymc.Exponential("slow", beta=treelen/minp)

    #Parameters:
    paramscales = [None]*(nparam-1)
    for p in range(nparam-1):
        paramscales[p] =  ShiftedGamma(name = "paramscale_{}".format(str(p)), shape = GAMMASHAPE, shift=1)

    mod = make_qmat_stoch(nobschar = nobschar,nregime=nregime,
                          nparam=nparam, mod_order_list = list(itertools.product(list(range(nparam+1)), repeat = nobschar**2-nobschar)),
                          modseed=modseed,
                          name="mod")

    # Likelihood function
    ar = hrm.create_hrm_ar(tree, chars, nregime, findmin=False)
    Q = np.zeros([nchar,nchar])

    @pymc.potential
    def mklik(mod=mod, slow=slow, paramscales=paramscales, ar=ar):
        np.copyto(ar["nodelist"],ar["nodelistOrig"])
        ar["root_priors"].fill(1.0)
        Qparams = [1e-15]*(nparam+1) # Initializing blank Qparams (maybe more efficient to do outside of function)
        s = slow
        Qparams[1] = slow
        for i,p in enumerate(paramscales):
            Qparams[i+2] = Qparams[i+1]*p

        mod_form = [mod[i*n_wr:i*n_wr+n_wr] for i in range(nregime)] + [mod[-n_br:]]
        fill_model_Q(mod_form, Qparams, Q)

        lik = hrm.hrm_mk(tree, chars, Q, nregime, pi=pi, returnPi=False,ar=ar)

        return lik
    if dbname is None:
        mod_mcmc = pymc.MCMC(locals(), calc_deviance=True)
    else:
        mod_mcmc = pymc.MCMC(locals(), calc_deviance=True, db="pickle",
                             dbname=dbname)
    mod_mcmc.use_step_method(QmatMetropolis, mod, nparam, nchar, nregime)
    return mod_mcmc



def fill_model_mk(mod, Qparams, Q, mask):
    """
    Fill cells of Q with Qparams based on mod
    """
    Q[mask] = [Qparams[i] for i in mod]
    Q[np.diag_indices(Q.shape[0])] = -np.sum(Q, 1)


def mk_allmodels_bayes(tree, chars, nparam, pi="Equal", dbname=None):
    """
    Fit an mk model with nparam parameters distributed about the Q matrix.
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = len(set(chars))
    ncell = nchar**2 - nchar
    assert nparam <= ncell

    minp = pscore(tree, chars)
    treelen = sum([n.length for n in tree.descendants()])
    ### Parameters
    # Prior on slowest distribution (beta = 1/mean)
    slow = pymc.Exponential("slow", beta=treelen/minp)


    paramscales = [None]*(nparam-1)
    for p in range(nparam-1):
        paramscales[p] =  pymc.Uniform(name = "paramscale_{}".format(str(p)), lower = 2, upper=20)
    ### Model
    paramset = list(range(nparam+1))
    nonzeros = paramset[1:]
    all_mods = list(itertools.product(paramset, repeat = ncell))
    all_mods = [tuple(m) for m in all_mods if all([i in set(m) for i in nonzeros])]

    mod = make_qmat_stoch_mk(all_mods, name="mod")

    l = mk.create_likelihood_function_mk(tree=tree, chars=chars, Qtype="ARD",
                                         pi=pi, findmin=False)
    Q = np.zeros([nchar, nchar])
    mask = np.ones([nchar,nchar], dtype=bool)
    mask[np.diag_indices(nchar)] = False
    @pymc.potential
    def mklik(mod=mod,slow=slow,paramscales=paramscales, name="mklik"):
        params = [0.0]*(nparam+1)
        params[1] = slow
        for i,s in enumerate(paramscales):
            params[2+i] = params[2+(i-1)] * s

        Qparams = [params[i] for i in mod]
        return l(np.array(Qparams))
    if dbname is None:
        mod_mcmc = pymc.MCMC(locals(), calc_deviance=True)
    else:
        mod_mcmc = pymc.MCMC(locals(), calc_deviance=True, db="pickle",
                             dbname=dbname)
    mod_mcmc.use_step_method(QmatMetropolis_mk, mod, all_mods)
    return mod_mcmc


def make_qmat_stoch_mk(all_mods, name="qmat_stoch"):
    """
    Stochastic for use in model-changing step
    """

    startingval = all_mods[0]

    @pymc.stochastic(dtype=tuple, name=name)
    def qmat_stoch(value=startingval):
        # Flat prior for all models
        return 0
    return qmat_stoch


class QmatMetropolis_mk(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new model
    """
    def __init__(self, stochastic, all_mods):
        pymc.Metropolis.__init__(self, stochastic, scale=1.)
        self.all_mods = all_mods
    def propose(self):
        cur_mod = self.stochastic.value
        new = random.choice([ m for m in self.all_mods if not m==cur_mod])
        self.stochastic.value = new

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value
