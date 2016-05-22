#!/usr/bin/env python
from __future__ import absolute_import, division, print_function, unicode_literals
import math
import itertools

import numpy as np
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
import pymc
import matplotlib.pyplot as plt

from ivy.chars.expokit import cyexpokit
from ivy.chars import mk, hrm, mk_mr

np.seterr(invalid="warn")


def create_mk_model(tree, chars, Qtype, pi):
    """
    Create model objects to be passed to pymc.MCMC

    Creates Qparams and likelihood function
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = len(set(chars))
    if Qtype=="ER":
        N = 1
    elif Qtype=="Sym":
        N = int(binom(nchar, 2))
    elif Qtype=="ARD":
        N = int((nchar ** 2 - nchar))
    else:
        ValueError("Qtype must be one of: ER, Sym, ARD")

    # Setting a Dirichlet prior with Jeffrey's hyperprior of 1/2
    if N != 1:
        theta = [1.0/2.0]*N
        Qparams_init = pymc.Dirichlet("Qparams_init", theta, value = [0.5])
        Qparams_init_full = pymc.CompletedDirichlet("Qparams_init_full", Qparams_init)
    else:
        Qparams_init_full = [[1.0]]

    # Exponential scaling factor for Qparams
    scaling_factor = pymc.Exponential(name="scaling_factor", beta=1.0, value=1.0)

    # Scaled Qparams; we would not expect them to necessarily add
    # to 1 as would be the case in a Dirichlet distribution
    @pymc.deterministic(plot=False)
    def Qparams(q=Qparams_init_full, s=scaling_factor):
        Qs = np.empty(N)
        for i in range(N):
            Qs[i] = q[0][i]*s
        return Qs

    l = mk.create_likelihood_function_mk(tree=tree, chars=chars, Qtype=Qtype,
                                  pi="Equal", findmin=False)
    @pymc.potential
    def mklik(q = Qparams, name="mklik"):
        return l(q)
    return locals()


def fit_mk_bayes(tree, chars, Qtype, pi, *kwargs):
    """
    Fit an mk model to a given tree and list of characters. Return
    posterior distributions of Q parameters and MAP estimate of Q matrix

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (list): List of character states corresponding to leaf nodes in
          preoder sequence. Character states must be in the form of 0,1,2,...
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
        Qtype: Either a string specifying how to esimate values for Q or a
          numpy array of a pre-specified Q matrix.

          Valid strings for Q:

          "Equal": All rates equal
          "Sym": Forward and reverse rates equal
          "ARD": All rates different
    Keyword Args:
        iters (float): Number of iterations in MCMC. Defaults to 2000
        burn (float): Burnin to discard. Defaults to 200
        thin (float): Thinning parameter. Defaults to 1

    Returns:
        tuple: The pymc MCMC object and the pymc MAP object
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nchar = len(set(chars))
    mod = create_mk_model(tree, chars, Qtype, pi)

    # Arguments for MCMC
    if not kwargs:
        kwargs = {}
    iters = kwargs.pop("iters", 2000)
    burn = kwargs.pop("burn", 200)
    thin = kwargs.pop("thin", 1)

    # MCMC samples
    mc = pymc.MCMC(mod)
    mc.sample(iters, burn, thin)

    # MAP estimation
    mp = pymc.MAP(mod)
    mp.fit()

    return (mc, mp)


def create_multi_mk_model(tree, chars, Qtype, pi, nregime=2):
    """
    Create an mk model with multiple regimes to be sampled from with MCMC.

    Regime number is fixed and the location of the regime shift is allowed
    to change
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    # Preparations
    nchar = len(set(chars))
    if Qtype=="ER":
        N = 1
    elif Qtype=="Sym":
        N = int(binom(nchar, 2))
    elif Qtype=="ARD":
        N = int((nchar ** 2 - nchar))
    else:
        ValueError("Qtype must be one of: ER, Sym, ARD")
    # This model has 2 components: Q parameters and a switchpoint
    # They are combined in a custom likelihood function

    ###########################################################################
    # Switchpoint:
    ###########################################################################
    # Modeling the movement of the regime shift(s) is the tricky part
    # Regime shifts will only be allowed to happen at a node
    # Regime shift: Uniform categorical distribution
    valid_switches = [i.ni for i in tree if not (i.isleaf or i.isroot)]
    # Uniform
    switch_ind = pymc.DiscreteUniform("switch_ind",lower=0, upper=len(valid_switches)-1)
    @pymc.deterministic(dtype=int)
    def switch(name="switch",switch_ind=switch_ind):
        return valid_switches[switch_ind]
    ###########################################################################
    # Qparams:
    ###########################################################################
    # Unscaled Q param: Dirichlet distribution
    # Setting a Dirichlet prior with Jeffrey's hyperprior of 1/2
    theta = [1.0/2.0]*N

    # One set of Q-parameters per regime
    allQparams_init = np.empty(nregime, dtype=object)
    allQparams_init_full = np.empty(nregime, dtype=object)
    allScaling_factors = np.empty(nregime, dtype=object)
    for i in range(nregime):
        if N != 1:
            allQparams_init[i] = pymc.Dirichlet("allQparams_init"+str(i), theta)
            allQparams_init_full[i] = pymc.CompletedDirichlet("allQparams_init_full"+str(i), allQparams_init[i])
        else: # Dirichlet function does not like creating a distribution
              # with only 1 state. Set it to 1 by hand
            allQparams_init_full[i] = [[1.0]]
        # Exponential scaling factor for Qparams
        allScaling_factors[i] = pymc.Exponential(name="allScaling_factors"+str(i), beta=1.0)
        # Scaled Qparams; we would not expect them to necessarily add
        # to 1 as would be the case in a Dirichlet distribution

    # Regimes are grouped by rows. Each row is a regime.
    @pymc.deterministic(plot=False)
    def Qparams(q=allQparams_init_full, s=allScaling_factors):
        Qs = np.empty([nregime,N])
        for n in range(N):
            for i in range(nregime):
                Qs[i][n] = q[i][0][n]*s[i]
        return Qs
    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function

    # Pre-allocating arrays
    qarray = np.zeros([nregime,N])
    locsarray = np.empty([2], dtype=object)
    l = mk_mr.create_likelihood_function_multimk(tree=tree, chars=chars,
        Qtype=Qtype,
        pi="Equal", findmin=False, nregime=2)

    @pymc.potential
    def multi_mklik(q = Qparams, switch=switch, name="multi_mklik"):

        locs = mk_mr.locs_from_switchpoint(tree,tree[int(switch)],locsarray)

        np.copyto(qarray, q)
        return l(qarray, locs=locs)
    return locals()


def nshifts(node, inds, n=0):
    """
    The number of regime shifts, given indices of regimes
    """
    if not node.children:
        return n
    elif node.isroot:
        n = nshifts(node.children[0], inds) + nshifts(node.children[1], inds)
        return n
    else:
        st = inds[node.ni-1]
        for i in node.children:
            if inds[i.ni-1] != st:
                n += 1
                n = nshifts(i, inds, n)
            else:
                n = nshifts(i,inds,n)
        return n


def get_indices(list_of_lists):
    """
    Get indices given list of nodes in regimes
    """
    indices = [0] * sum(map(len, list_of_lists))

    for listno, alist in enumerate(list_of_lists):
            for n in alist:
                indices[n-1] = listno
    return indices


def mk_results(mcmc_obj):
    """
    Create summary graphs and statistics for an mk model

    Args:
        mcmc_obj: A pymc MCMC object that has been sampled
    Returns:
        dict: Trace plots, histograms, and summary statistics
    """
    varsToPlot = ["Qparams"]
    if "switch" in [ i.__name__ for i in mcmc_obj.variables ]:
        varsToPlot.append("switch")

    traces = {}
    hists = {}
    traceplots = {}
    summary = {}

    for var in varsToPlot:
        if var == "Qparams":
            traces[var] = {}
            hists[var] = {}
            traceplots[var] = {}
            summary[var] = {}
            for i in range(mcmc_obj.trace(var)[:].shape[1]):
                if "switch" in varsToPlot:
                    for k in range(mcmc_obj.trace(var)[:].shape[2]):
                        traces[var][str([i,k])] = [ m[i][k] for m in mcmc_obj.trace(var)[:] ]
                        hists[var][str([i,k])] = plt.hist(traces[var][str([i,k])])
                        traceplots[var][str([i,k])] = plt.plot(traces[var][str([i,k])])
                        summary[var][str([i,k])] = np.percentile(traces[var][str([i,k])], [2.5, 50, 97.5])
                else:
                    traces[var][str(i)] = [ m[i] for m in mcmc_obj.trace(var)[:]]
                    hists[var][str(i)] = plt.hist(traces[var][str(i)])
                    traceplots[var][str(i)] = plt.plot(traces[var][str(i)])
                    summary[var][str(i)] = np.percentile(traces[var][str(i)], [2.5, 50, 97.5])
        else:
            traces[var] = [ m for m in mcmc_obj.trace(var)[:] ]
            hists[var] = plt.hist(traces[var])
            traceplots[var] = plt.plot(traces[var])
            summary[var] = scipy.stats.mode(traces[var])[0]
    return {"traces":traces, "hists":hists, "traceplots":traceplots, "summary":summary}


def hrm_bayesian(tree, chars, Qtype, nregime, pi="Fitzjohn", constraint="Rate"):
    """
    Create a hidden rates model for pymc to be sampled from.

    Args:
        tree (Node): Root node of a tree. All branch lengths must be
          greater than 0 (except root)
        chars (dict): Dict mapping character states to tip labels.
          Character states should be coded 0,1,2...

          Can also be a list with tip states in preorder sequence
        pi (str): Either "Equal", "Equilibrium", or "Fitzjohn". How to weight
          values at root node. Defaults to "Equal"
          Method "Fitzjohn" is not thouroughly tested, use with caution
        Qtype: Either a string specifying how to esimate values for Q or a
          numpy array of a pre-specified Q matrix.
            "Simple": Symmetric rates within observed states and between
              rates.
            "STD": State Transitions Different. Transitions between states
              within the same rate class are asymetrical
        nregime (int): Number of hidden states. nstates = 0 is
          equivalent to a vanilla Mk model
        constraint (str): Contraints to apply to the parameters of the Q matrix.
          Can be one of the following:
            "Rate": The fastest rate in the fastest regime must be faster than
              the fastest rate in the slowest regime
            "Symmetry": For two-regime models only. The two regimes must
              have different symmetry (a>b in regime 1, b>a in regime 2)
            "None": No contraints
    """
    if type(chars) == dict:
        chars = [chars[l] for l in [n.label for n in tree.leaves()]]
    nobschar = len(set(chars))
    nchar = nobschar * nregime
    assert Qtype in ["Simple", "STD", "RTD", "ARD"], "Q type must be one of: simple, STD, RTD, ARD"
    ###########################################################################
    # Qparams:
    ###########################################################################
    # The simple model has # of parameters equal to nregime + 1 (One set of
    # rates for each regime, plus transition rate between regimes)
    # For now, we will have each be exponentially distributed

    # Simplest model: all transitions between states within a regime are equal.
    # Each regime has a rate associated with it.
    # There is one rate between regimes.
    # Number of parameters = nregime+1
    if Qtype == "Simple":
        # Wtihin-regime transitions
        WR_Qparams = np.ndarray(nregime, dtype="object")
        for i in range(nregime):
            WR_Qparams[i] = pymc.Exponential(name="wr-par"+str(i), beta = 1.0, value = 1e-2+(i/100.0))
        # Between-regime transitions:
        BR_Qparams = pymc.Exponential(name="br-par", beta = 1.0, value = 1e-2)
    # State-transitions different. Transitions between states within a
    # rate regime can differ (ARD). Transitions between rates share one
    # rate parameter.
    # Number of parameters = nregime*(nobschar**2-nobschar) + 1
    if Qtype == "STD":
        theta = [1.0/2.0] * nobschar
        i_d = np.ndarray(nregime, dtype="object")
        c_d = np.ndarray(nregime, dtype="object")
        scale = np.ndarray(nregime, dtype="object")
        # Within-regime transitions
        WR_Qparams = np.ndarray(nregime, dtype="object")
        for i in range(nregime):
            # First, create a dirichlet distribution
            i_d[i] = pymc.Dirichlet("parInit_"+str(i), theta, value = [1.0/nobschar]*(nobschar-1))
            c_d[i] = pymc.CompletedDirichlet("parInit"+str(i), i_d[i])
            scale[i] = pymc.Exponential(name="scaling"+str(i), beta=1.0, value=1e-2+(i/100.0))
            # Then, scale dirichlet distribution by overall rate parameter for that regime
            @pymc.deterministic(plot=False,name="wr-par"+str(i))
            def d_scaled(d = c_d[i], s = scale[i]):
                return (d*s)[0]
            WR_Qparams[i] = d_scaled
        # Between-regime transitions
        BR_Qparams = pymc.Exponential(name="br-par", beta = 1.0, value = 1e-2)
    if Qtype == "RTD":
        WR_Qparams = np.ndarray(nregime, dtype="object")
        for i in range(nregime):
            WR_Qparams[i] = pymc.Exponential(name="wr-par"+str(i), beta = 1.0, value = 1e-2+(i/100.0))

        BR_Qparams = np.ndarray(nregime-1, dtype="object")
        for i in range(nregime-1):
            BR_Qparams[i] = pymc.Exponential(name="br-par"+str(i), beta=1.0, value=1e-2)
    if Qtype == "ARD":
        theta = [1.0/2.0] * nobschar
        i_d = np.ndarray(nregime, dtype="object")
        c_d = np.ndarray(nregime, dtype="object")
        scale = np.ndarray(nregime, dtype="object")
        # Within-regime transitions
        WR_Qparams = np.ndarray(nregime, dtype="object")

        for i in range(nregime):
            # First, create a dirichlet distribution
            i_d[i] = pymc.Dirichlet("parInit_"+str(i), theta, value = [1.0/nobschar]*(nobschar-1))
            c_d[i] = pymc.CompletedDirichlet("parInit"+str(i), i_d[i])

            scale[i] = pymc.Exponential(name="scaling"+str(i), beta=1.0, value=1e-2+(i/100.0))
            # Then, scale dirichlet distribution by overall rate parameter for that regime
            @pymc.deterministic(plot=False,name="wr-par"+str(i))
            def d_scaled(d = c_d[i], s = scale[i]):
                return (d*s)[0]
            WR_Qparams[i] = d_scaled
        BR_Qparams = np.ndarray((nregime-1)*2*nobschar, dtype="object")
        br_i = 0
        for i in list(range(nregime-1))*2:
            for n in range(nobschar):
                BR_Qparams[br_i] = pymc.Exponential(name="br-par"+str(br_i), beta=1.0, value=1e-2)
                br_i += 1
    ###########################################################################
    # Likelihood
    ###########################################################################
    l = hrm.create_likelihood_function_hrm_mk(tree=tree, chars=chars,
        nregime=nregime, Qtype="ARD", pi=pi, findmin=False)
    @pymc.potential
    def mklik(wr = WR_Qparams, br=BR_Qparams, name="mklik"):
        if Qtype == "Simple":
            # Getting the locations of each Q parameter to feed
            # to the likelihood function

            # Note that the likelihood function takes q parameters
            # in coumnwise-order, not counting zero and negative values.
            # Within-regime shifts
            qinds = {}
            for i,q in enumerate(wr):
                qinds[i]=valid_indices(nobschar, nregime, i,i)
            rshift_pairs = list(zip(list(range(nregime))[1:], list(range(nregime))[:-1]))
            qinds[i+1] = [] # Between-regime shifts(all share 1 rate)
            for p in rshift_pairs:
                qinds[i+1].extend(valid_indices(nobschar, nregime, p[0],p[1]))
                qinds[i+1].extend(valid_indices(nobschar, nregime, p[1],p[0]))
            # These are the indices of the values we will give to
            # the likelihood function, in order
            param_indices = sorted([ i for v in list(qinds.values()) for i in v])
            qparam_list = list(wr)+[br] # Making a single list to get parameters from
            Qparams = [] # Empty list for values to feed to lik function
            for pi in param_indices:
                qi = [ k for k,v in qinds.items() if pi in v ][0]
                Qparams.append(qparam_list[qi]) # Pulling out the correct param
            # Qparams now contains the parameters needed in the
            # correct order for the likelihood function.
            if constraint == "Rate":
                if ((sorted(list(wr)) == list(wr)) and (br < wr[nregime-1])):
                    return l(np.array(Qparams))
                else:
                    return -np.inf
            else:
                return l(np.array(Qparams))
        if Qtype == "STD":
            qinds = {}
            n=0
            for i,q in enumerate(wr):
                for k in range(nregime):
                    qinds[n] = [valid_indices(nobschar, nregime, i, i)[k]]
                    n+=1
            rshift_pairs = list(zip(list(range(nregime))[1:], list(range(nregime))[:-1]))
            qinds[n] = [] # Between-regime shifts(all share 1 rate)
            for p in rshift_pairs:
                qinds[n].extend(valid_indices(nobschar, nregime, p[0],p[1]))
                qinds[n].extend(valid_indices(nobschar, nregime, p[1],p[0]))
            param_indices = sorted([ i for v in list(qinds.values()) for i in v])
            qparam_list = [i for s in [q for q in wr] for i in s]+[br] # Making a single list to get parameters from
            Qparams = [] # Empty list for values to feed to lik function
            for pi in param_indices:
                qi = [ k for k,v in qinds.items() if pi in v ][0]
                Qparams.append(qparam_list[qi]) # Pulling out the correct param
            # Potential constraints are "Rate" and "Symmetry"
            if constraint == "Rate":
                for i in range(nregime):
                    n = [q[i] for q in wr]
                    if not sorted(n) == n:
                        return -np.inf
            if constraint == "Symmetry":
                assert nchar == 4
                for i in range(nregime):
                    if not (wr[0][0]/wr[0][1] <= 1) and (wr[1][0]/wr[1][1] >= 1):
                        return -np.inf
            if br > max(wr[nregime-1]):
                return -np.inf

            return l(np.array(Qparams))
        if Qtype == "RTD":
            raise AssertionError
            qinds = {}
            for i,q in enumerate(wr):
                qinds[i]=valid_indices(nobschar, nregime, i,i)
        if Qtype == "ARD":
            qinds = {}
            n=0
            for i,q in enumerate(wr):
                for k in range(nregime):
                    qinds[n] = [valid_indices(nobschar, nregime, i, i)[k]]
                    n+=1
            rshift_pairs = list(zip(list(range(nregime))[1:], list(range(nregime))[:-1]))
            for p in rshift_pairs:
                for i in  valid_indices(nobschar, nregime, p[0],p[1]):
                    qinds[n] = [i]
                    n+=1
                for i in  valid_indices(nobschar, nregime, p[1],p[0]):
                    qinds[n] = [i]
                    n+=1
            param_indices = sorted([ i for v in list(qinds.values()) for i in v])
            qparam_list = [i for s in [q for q in wr] for i in s]+[b for b in br] # Making a single list to get parameters from
            Qparams = [] # Empty list for values to feed to lik function
            for pi in param_indices:
                qi = [ k for k,v in qinds.items() if pi in v ][0]
                Qparams.append(qparam_list[qi]) # Pulling out the correct param
            for i in range(nregime):
                n = [q[i] for q in wr]
                if not sorted(n) == n:
                    return -np.inf
            if max(br) > max(wr[nregime-1]):
                return -np.inf
            return l(np.array(Qparams))
    return locals()


def _subarray_indices(nobschar, nregime, x, y):
    """
    For a Q matrix whose elements are to be indexed columnwise,
    return the indices of the elements of the given subarray (array
    divided into nregime x nregime subarrays)
    """
    xinds = [ i + x*nobschar for i in range(nobschar) ]
    yinds = [ i + y*nobschar for i in range(nobschar) ]
    ylen = nobschar*nregime
    sub = [x + y for y in [c* ylen for c in yinds] for x in xinds ]
    return sub


def _invalid_indices(nobschar, nregime):
    # Diagonals have no parameters
    diags = [ n + nobschar*nregime*n for n in range(nobschar*nregime)]
    consecs = list(zip(list(range(nregime))[1:], list(range(nregime))[:-1]))
    revconsecs = [ i[::-1] for i in consecs ]

    subarrays = list(itertools.permutations(list(range(nregime)), 2))
    # "corner" subarrays have no parameters (no shifts directly from
    # slow to fast regimes, eg.)
    corner_subarrays = [ i for i in subarrays if not i in consecs+revconsecs ]
    corner_indices = list(range(len(corner_subarrays)))
    for i,s in enumerate(corner_subarrays):
        corner_indices[i] = _subarray_indices(nobschar, nregime, s[0], s[1])

    # For regime-shift subarrays (non-diagonal and non-corner subarrays)
    # Only diagonals are allowed. All non-diagonals have no parameters.
    rshift_subarrays = [ i for i in subarrays if i in consecs+revconsecs]
    rshift_indices = list(range(len(rshift_subarrays)))
    for i,s in enumerate(rshift_subarrays):
        inds = _subarray_indices(nobschar, nregime, s[0], s[1])
        # Remove the diagonals (they are valid)
        inds = [ inds[r] for r in range(len(inds)) if not r in range(0, nobschar**2, nobschar)]
        rshift_indices[i] = inds
    return [ i for sub in [[diags] + corner_indices + rshift_indices][0] for i in sub ]


def valid_indices(nobschar, nregime, x, y):
    """
    Return only the valid indices for the given subarray
    """
    l = _subarray_indices(nobschar, nregime, x, y)
    inv = _invalid_indices(nobschar, nregime)

    return [ i for i in l if not i in inv]
