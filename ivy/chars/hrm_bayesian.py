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

from pymc.distributions import negative_binomial_like as nbl


def unique_models(nchar, nregime, nparams):
    """
    Create a list of all possible models for a Q matrix with nchar distinct
    character states and nregime distinct regimes, filling all cells
    with nparams + 0.

    Args:
        nchar (int): Number of observed characters
        nregime (int): Number of regimes in model
        nparams (int): Number of unique parameters in model (not including 0)

    Returns:
        list: List of all unique models. Each item contains a tuple of the
          within-regime parameters and a tuple of the between-regime parameters.
    """
    param_ranks = range(nparams+1)

    # Within-regime parameters
    distinct_regimes = list(itertools.permutations(param_ranks,nchar))
    for p in param_ranks:
        distinct_regimes.append((p,p))
    wr = list(itertools.combinations(distinct_regimes,nregime))
    wr_flat = [ tuple([i for s in  x for i in s]) for x in wr ]

    # Between-regime parameters
    n_br = (nregime**2 - nregime)*nchar
    br = list(itertools.product(param_ranks, repeat = n_br))

    mods = list(itertools.product(wr_flat, br))

    # Removing redundant regimes
    mods_flat = [ tuple([i for s in  x for i in s]) for x in mods ]
    regime_identities = [make_identity_regime(m) for m in mods_flat]

    unique_mods = [mods_flat[mods_flat.index(i)] for i in set(regime_identities)]

    return unique_mods


def make_identity_regime(mod):
    """
    Given one model, reduce it to its identity (eg (2,2),(3,3) becomes
    (1,1),(2,2)
    """
    params_present = list(set(mod))
    if 0 in params_present:
        identity_params = range(len(params_present))
    else:
        identity_params = [i+1 for i in range(len(params_present))]
    identity_regime =  tuple([identity_params[params_present.index(i)]
                                     for i in mod])
    return identity_regime

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

def make_qmat_stoch(graph):
    @pymc.stochastic(dtype=tuple)
    def qmat_stoch(value = mod):
        ln = sum([nbl(i, mu = 1, alpha = 1) for i in mod])
        return ln
    return qmat_stoch

class QmatMetropolis(pymc.Metropolis):
    def __init__(self, stochastic, graph):
        pymc.Metropolis.__init__(self, stochastic, scale=1.)
        self.graph = graph
    def propose(self):
        mod = self.stochastic.value

        connected = self.graph[mod].keys()

        new = random.choice(connected)

        self.stochastic.value = new

    def reject(self):
        self.rejected += 1
        self.stochastic.value = self.stochastic.last_value

qs = make_qmat_stoch(mod_graph)

mod_mc = pymc.MCMC([qs])
mod_mc.use_step_method(QmatMetropolis, qs, graph=mod_graph)

mod_mc.sample(100000)


visits = [tuple(i) for i in mod_mc.trace("qmat_stoch")[:]]
Counter(visits).most_common()
