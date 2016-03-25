"""
Functions for testing an hrm model by assigning parameters to different
cells of the Q matrix. Use custom step methods in pymc to traverse all
possible models.
"""
import ivy
import numpy as np
import math
import itertools
import pymc
import random
import collections
import networkx as nx
import scipy

from ivy.chars.anc_recon import find_minp
from ivy.chars import hrm

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
    param_ranks = range(nparam+1)
    n_wr = nchar**2 - nchar

    # Within-regime parameters
    distinct_regimes = list(itertools.permutations(param_ranks,n_wr))
    for p in param_ranks:
        distinct_regimes.append((p,p))
    wr = list(itertools.combinations(distinct_regimes,nregime))
    wr_flat = [ tuple([i for s in  x for i in s]) for x in wr ]

    # Between-regime parameters
    n_br = (nregime**2 - nregime)*nchar
    # Disallow the fasted parameter in between-regime transitions
    br = list(itertools.product(param_ranks[:-1], repeat = n_br))
    br.pop(0) # Need to handle "no transition" models separately

    mods = list(itertools.product(wr_flat, br))

    # "no transition" models
    nt_wr = [i+tuple([0]*n_wr) for i in distinct_regimes if set(i) != {0} ]
    nt_mods = [ (i, tuple([0]*n_br)) for i in nt_wr ]
    mods.extend(nt_mods)

    mods_flat = [ tuple([i for s in  x for i in s]) for x in mods ]

    return mods_flat

def make_model_graph(unique_mods):
    """
    Create graph of models where each model is adjacent to all other models
    that differ from it by one parameter

    Args:
        unique_mods (list): List of models (must be tuples). Identical to output
          of unique_models
    Returns:
        Graph: Nx graph of all models
    """
    mod_graph = nx.Graph()
    mod_graph.add_nodes_from(unique_mods)

    # Adding edges
    # Iterate over all pairs and determine if they form an edge
    for i,mod in enumerate(unique_mods):
        for n in unique_mods[i+1:]:
            if sum([mod[i] != n[i] for i in range(len(mod))]) == 1:
                mod_graph.add_edge(mod, n)
    return mod_graph

def make_qmat_stoch(graph, name="qmat_stoch"):
    """
    Make a stochastic to use for a model-changing step
    """
    startingval = (1,0,0,0,0,0,0,0)
    nmodel = len(graph.node)
    nsingle = len([ i[-4:] for i in graph.node.keys() if set(i[-4:]) == {0}])

    one_regime_li = 0.5/nsingle
    multi_regime_li = 0.5/(nmodel-nsingle)

    @pymc.stochastic(dtype=tuple, name=name)
    def qmat_stoch(value = startingval):
        # Very simple prior: single-regime models should take up half
        # of likelihood space
        if set(value[-4:]) == {0}:
            ln_li = np.log(one_regime_li)
        else:
            ln_li = np.log(multi_regime_li)
        return ln_li
    return qmat_stoch

class QmatMetropolis(pymc.Metropolis):
    """
    Custom step algorithm for selecting a new model
    """
    def __init__(self, stochastic, graph):
        pymc.Metropolis.__init__(self, stochastic, scale=1.)
        self.graph = graph
    def propose(self):
        cur_mod = self.stochastic.value
        connected = self.graph[cur_mod].keys()
        new = random.choice(connected)
        self.stochastic.value = new

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value



def ShiftedGamma(shape, shift = 1, name="ShiftedGamma"):
    @pymc.stochastic(name=name)
    def shifted_gamma(value=2, shape=shape):
        return pymc.gamma_like(value-shift, shape, 1)
    return shifted_gamma

def hrm_allmodels_bayes(tree, chars, nregime, nparam, pi="Equal", mod_graph=None):
    """
    Use an MCMC chain to fit a hrm model with a limited number of parameters.

    The chain will step through different models, placing a prior on simpler
    models


    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        nregime (int): Number of regimes
        nparam (int): Number of unique parameters to allow in a model
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
        mod_graph (Graph): Nx graph of all possible models. Optional, if not
          provided will generate graph, which may be time-consuming
    """
    nobschar = len(set(chars))
    nchar = nobschar * nregime
    n_wr = nobschar**2-nobschar
    n_br = (nregime**2-nregime)*nobschar

    if mod_graph is None:
        mod_graph = make_model_graph(unique_models(nobschar, nregime, nparam))

    minp = find_minp(tree, chars)
    treelen = sum([n.length for n in tree.descendants()])
    # Prior on slowest distribution (beta = 1/mean)
    slow = pymc.Exponential("slow", beta=treelen/minp)

    #TODO generalize
    gamma_shape = 3.55
    alpha = ShiftedGamma(name = "alpha", shape = gamma_shape, shift=1)
    beta = ShiftedGamma(name = "beta", shape = gamma_shape, shift=1)

    mod = make_qmat_stoch(mod_graph, name="mod")

    # Likelihood function
    ar = hrm.create_hrm_ar(tree, chars, nregime, findmin=False)
    Q = np.zeros([nchar,nchar])

    @pymc.potential
    def mklik(mod=mod, slow=slow, alpha=alpha, beta=beta, ar=ar):
        np.copyto(ar["nodelist"],ar["nodelistOrig"])
        ar["root_priors"].fill(1.0)
        s = slow
        m = s*alpha
        f = m*beta
        Qparams = [1e-15, s, m, f]

        mod_form = [mod[i*n_wr:i*n_wr+n_wr] for i in range(nregime)] + [mod[-n_br:]]
        fill_model_Q(mod_form, Qparams, Q)

        lik = hrm.hrm_mk(tree, chars, Q, nregime, pi=pi, returnPi=False,ar=ar)

        return lik
    mod_mcmc = pymc.MCMC(locals(), calc_deviance=True)
    mod_mcmc.use_step_method(QmatMetropolis, mod, graph=mod_graph)
    return mod_mcmc

def fill_model_Q(mod, Qparams, Q):
    """
    Create Q matrix given model and parameters.

    Args:
        mod (list): List of tuples. Code for the mode used. The first nregime
          tuples correspond to the within-regime transition rates for each
          regime. The last tuple corresponds to the between-regime transition
          rates. Ex. [(2, 1), (0, 0), (0, 0, 1, 0)] corresponds to a
          matrix of:
              [-,2,0,-]
              [1,-,-,0]
              [1,-,-,0]
              [-,0,0,-]
        Qparams (list): List of floats corresponding to the values for
          slow, medium, fast, etc. rates. The first is always 0
        Q (np.array): Pre-allocated Q matrix
    """
    nregime = len(mod)-1
    nobschar = len(mod[0])
    nchar = nregime*nobschar
    # TODO: generalize
    Q.fill(0.0)
    for i in range(nregime):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        np.copyto(Q[subQ, subQ], [[0, Qparams[mod[i][0]]],[Qparams[mod[i][1]], 0]])

    for i,submatrix_index in enumerate(itertools.permutations(range(nregime),2)):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        nregimeswitch =(nregime**2 - nregime)*2
        np.fill_diagonal(Q[my_slice0,my_slice1],[Qparams[p] for p in mod[nregime][i*nobschar:i*nobschar+nobschar]])
    np.fill_diagonal(Q, -np.sum(Q,1))


def lognormal_percentile(v1, v2, p):
    """
    Return mu and tau for a lognormal distribution where
    p percent of the distribution falls between v1 and v2
    """
    mn = np.mean([np.log(v1),np.log(v2)])
    SD = (np.log(v2) - mn) / scipy.stats.norm.ppf(p+(1-p)/2)
    return mn, SD
