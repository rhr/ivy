import ivy
import numpy as np
import math
from ivy.chars.expokit import cyexpokit
import scipy
from scipy import special
from scipy.optimize import minimize
from scipy.special import binom
from ivy.chars import discrete
import pymc

def create_mk_model(tree, chars, Qtype, pi):
    """
    Create model objects to be passed to pymc.MCMC

    Creates Qparams and likelihood function
    """
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
        Qparams_init = pymc.Dirichlet("Qparams_init", theta)
        Qparams_init_full = pymc.CompletedDirichlet("Qparams_init_full", Qparams_init)
    else:
        Qparams_init_full = [[1.0]]

    # Exponential scaling factor for Qparams
    scaling_factor = pymc.Exponential(name="scaling_factor", beta=1.0)

    # Scaled Qparams; we would not expect them to necessarily add
    # to 1 as would be the case in a Dirichlet distribution
    @pymc.deterministic(plot=False)
    def Qparams_scaled(q=Qparams_init_full, s=scaling_factor):
        Qs = np.empty(N)
        for i in range(N):
            Qs[i] = q[0][i]*s
        return Qs

    l = discrete.create_likelihood_function_mk(tree=tree, chars=chars, Qtype=Qtype,
                                  pi="Equal", min=False)
    @pymc.potential
    def mklik(q = Qparams_scaled, name="mklik"):
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
        Q: Either a string specifying how to esimate values for Q or a
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
            allQparams_init_full[i] = pymc.CompletedDirichlet("allQparams_init_full"+str(i), Qparams_init[i])
        else: # Dirichlet function does not like creating a distribution
              # with only 1 state. Set it to 1 by hand
            allQparams_init_full[i] = [[1.0]]
        # Exponential scaling factor for Qparams
        allScaling_factors[i] = pymc.Exponential(name="allScaling_factors"+str(i), beta=1.0)
        # Scaled Qparams; we would not expect them to necessarily add
        # to 1 as would be the case in a Dirichlet distribution
    @pymc.deterministic(plot=False)
    def Qparams_scaled(q=allQparams_init_full, s=allScaling_factors):
        Qs = np.empty([N, nregime])
        for n in range(N):
            for i in range(nregime):
                Qs[n][i] = q[i][0][n]*s[i]
        return Qs
    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function

    # Pre-allocating arrays
    qarray = np.zeros([N,nregime])
    locsarray = np.empty([2], dtype=object)

    print "generating likelihood function"
    l = discrete.create_likelihood_function_multimk_b(tree=tree, chars=chars,
        Qtype=Qtype,
        pi="Equal", min=False, nregime=2)

    @pymc.potential
    def multi_mklik(q = Qparams_scaled, switch=switch, name="multi_mklik"):

        locs = discrete.locs_from_switchpoint(tree,tree[int(switch)],locsarray)

        # l = discrete.create_likelihood_function_multimk(tree=tree, chars=chars,
        #     Qtype=Qtype, locs = locs,
        #     pi="Equal", min=False)
        np.copyto(qarray, q)
        return l(qarray[0], locs)
    return locals()


def create_multi_mk_model_2(tree, chars, Qtype, pi, nregime=2):
    """
    Create an mk model with multiple regimes to be sampled from with MCMC.

    Allows multiple switches between regimes

    Regime number is fixed and the location of the regime shift is allowed
    to change
    """
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
    # Regime locations
    ###########################################################################

    # Sampling without replacement from a vector into N bins, where
    # N is the number of regimes
    def split_into_bins(nbin, vector):
        """
        Randomly split vector into nbin number of bins, each of random size
        """
        permutation = list(np.random.permutation(vector))

        splits = sorted(np.random.choice(range(1,len(vector)), nbin-1, replace=False))

        bins = [[]]*nbin

        start = 0
        end = splits[0]

        for i in range(nbin):
            bins[i] = permutation[start:end]
            start = end
            try:
                end = splits[i+1]
            except IndexError:
                end = len(vector)

        return bins

    @pymc.stochastic(dtype=object)
    def regimeIndices(value = np.array([[1],range(2,(len(tree.descendants())+1))], dtype=object), nbin=nregime, vector=range(1,len(tree.descendants())+1)):
        def logp(nbin, vector, value):
            return np.log(1.0/len(tree.descendants()))
        def random(nbin, vector):
            return np.array(split_into_bins(nbin, vector))

    # Counting the number of steps involved in a regime
    inds = np.zeros(len(tree.descendants()))
    def get_indices(tree, locs, inds=inds):
        for i,n in enumerate(tree.descendants()):
            inds[i] =[ j for j,l in enumerate(locs) if n.ni in l ][0]
        return inds
    def nshifts(node, inds, n=0):
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
    @pymc.deterministic(dtype=int)
        def nswitches(regimeIndices, tree):
            inds = get_indices(tree, regimeIndices, inds=inds)
            return nshifts(tree, inds)

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
            allQparams_init_full[i] = pymc.CompletedDirichlet("allQparams_init_full"+str(i), Qparams_init[i])
        else: # Dirichlet function does not like creating a distribution
              # with only 1 state. Set it to 1 by hand
            allQparams_init_full[i] = [[1.0]]
        # Exponential scaling factor for Qparams
        allScaling_factors[i] = pymc.Exponential(name="allScaling_factors"+str(i), beta=1.0)
        # Scaled Qparams; we would not expect them to necessarily add
        # to 1 as would be the case in a Dirichlet distribution
    @pymc.deterministic(plot=False)
    def Qparams_scaled(q=allQparams_init_full, s=allScaling_factors):
        Qs = np.empty([N, nregime])
        for n in range(N):
            for i in range(nregime):
                Qs[n][i] = q[i][0][n]*s[i]
        return Qs
    ###########################################################################
    # Likelihood
    ###########################################################################
    # The likelihood function

    # Pre-allocating arrays
    qarray = np.zeros([N,nregime])
    l = discrete.create_likelihood_function_multimk_b(tree=tree, chars=chars,
        Qtype=Qtype,
        pi="Equal", min=False, nregime=2)

    @pymc.potential
    def multi_mklik(q = Qparams_scaled, locs=regimeIndices, name="multi_mklik"):

        # l = discrete.create_likelihood_function_multimk(tree=tree, chars=chars,
        #     Qtype=Qtype, locs = locs,
        #     pi="Equal", min=False)
        np.copyto(qarray, q)
        return l(qarray[0], locs)
    return locals()


def nshifts(node, inds, n=0):
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

# def get_indices(list_of_lists):
#     indices = [0] * sum(map(len, list_of_lists))
#
#     for listno, alist in enumerate(list_of_lists):
#             for n in alist:
#                 indices[n-1] = listno
#     return indices

inds = np.zeros(len(tree.descendants()))
def get_indices(tree, locs, inds=inds):
    for i,n in enumerate(tree.descendants()):
        inds[i] =[ j for j,l in enumerate(locs) if n.ni in l ][0]
    return inds
