
Performing analyses
===================

``ivy`` has many tools for performing analyses on trees. Here we will cover
a few analyses you can perform.

Phylogenetically Independent Contrasts
--------------------------------------

You can perform PICs using ``ivy``'s ``PIC`` function. This function takes a
root node and a dictionary mapping node labels to character traits as inputs
and outputs a dictionary mappinginternal nodes to tuples containing ancestral
state, its variance (error), the contrast, and the contrasts's variance.

.. TODO:: Add citation for tree

The following example uses a consensus tree from Sarkinen et al. 2013 and
Ziegler et al. unpub. data.

Note: This function requires that the root node have a length that is not none.
Note: this function currently cannot handle polytomies.

.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: import csv
    In [*]: import matplotlib.pyplot as plt
    In [*]: r = ivy.tree.read("examples/solanaceae_sarkinen2013.newick")
    In [*]: r.length = 0
    In [*]: polvol = {}; stylen = {}
    In [*]: with open("examples/pollenvolume_stylelength.csv", "r") as csvfile:
                traits = csv.DictReader(csvfile, delimiter = ",", quotechar = '"')
                for i in traits:
                    polvol[i["Binomial"]] = float(i["PollenVolume"])
                    stylen[i["Binomial"]] = float(i["StyleLength"])

    In [*]: p = ivy.contrasts.PIC(r, polvol) # Contrasts for log-transformed pollen volume
    In [*]: s = ivy.contrasts.PIC(r, stylen) # Contrasts for log-transformed style length
    In [*]: pcons, scons = zip(*[ [p[key][2], s[key][2]] for key in p.keys() ])
    In [*]: plt.scatter(scons,pcons)
    In [*]: plt.show()


Lineages Through Time
---------------------

``ivy`` has functions for computing LTTs. The ``ltt`` function takes a root
node as input and returns a tuple of 1D-arrays containing the results of
times and diverisities.

Note: The tree is expected to be an ultrametric chromogram (extant leaves,
branch lengths proportional to time).

.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: r = ivy.tree.read("examples/solanaceae_sarkinen2013.newick")
    In [*]: v = ivy.ltt(r)

You can plot your results using ``matplotlib``.


.. sourcecode:: ipython

    In [*]: import matplotlib.pyplot as plt
    In [*]: plt.step(v[0], v[1])
    In [*]: plt.xlabel("Time"); plt.ylabel("Lineages"); plt.title("LTT")
    In [*]: plt.show()

.. image:: _images/ltt.png
    :width: 700


Phylorate plot
--------------

By accessing R libraries using `rpy2 <http://rpy.sourceforge.net/>`_, we can use
the functions in the `BAMMtools <https://cran.r-project.org/web/packages/BAMMtools/index.html>`_
R library to generate phylorate plots.

The following analysis is done using the whales dataset provided with BAMMtools.

.. sourcecode:: ipython

    In [*]: from ivy.r_funcs import phylorate
    In [*]: e = "whaleEvents.csv" # Event data created with BAMM
    In [*]: treefile = "whales.tre"
    In [*]: rates = phylorate(e, treefile, "netdiv")

We can add the results as a layer to a plot using the ``add_phylorate`` function
in ``ivy.vis.layers``

.. sourcecode:: ipython

    In [*]: from ivy.vis import layers
    In [*]: r = readtree(treefile)
    In [*]: fig = treefig(r)
    In [*]: fig.add_layer(layers.add_phylorate, rates[0], rates[1], ov=False,
           store="netdiv")



.. image:: _images/phylorate_plot.png
    :width: 700

Mk models
---------
``ivy`` has functions to fit an Mk model given a tree and a list of character
states. There are functions to fit the Mk model using both maximum likelihood
and Bayesian MCMC methods.

To fit an Mk model, you need a tree and a list of character states. This list
should only contain integers 0,1,2,...,N, with each integer corresponding to
a state. The list of characters should be provided in preorder sequence.

Let's read in some example data: plant habit in tobacco. We can load in a
csv containing binomials and character states using the ``loadChars`` function.
This gives us a dictionary mapping binomials to character states.

.. sourcecode:: ipython

    In [*]: tree = ivy.tree.read("examples/nicotiana.newick")
    In [*]: chars = ivy.tree.load_chars("examples/nicotianaHabit.csv")

Let's get our data into the correct format: we need to convert `chars` into
a list of 0's and 1's matching the character states in preorder sequence

.. sourcecode:: ipython

    In [*]: charsPreorder = [ chars[n.label]["Habit"] for n in tree.leaves() ]
    In [*]: charList = map(lambda x: 0 if x=="Herb" else 1, charsPreorder)

We can take a look at how the states are distributed on the tree using the
``tip_chars`` method on a tree figure object. In this case "Herb" will be
represented by green and "Shrub" will be represented by brown.

.. sourcecode:: ipython

    In [*]: fig = ivy.vis.treevis.TreeFigure(tree)
    In [*]: fig.tip_chars(charList, colors=["green", "brown"])

.. image:: _images/nicotiana_1.png
    :width: 700

Now we are ready to fit the model. We will go over the maximum likelihood
approach first.

Maximum Likelihood
~~~~~~~~~~~~~~~~~~
Perhaps the simplest way to fit an Mk model is with the maximum likelihood
approach. We will make use of the ``optimize`` module of ``scipy`` to find
the maximum likelihood values of this model.

First, we must consider what type of model to fit. `ivy` allows you to
specify what kind of instantaneous rate matrix (Q matrix) to use.
The options are:

* **"ER"**: An equal-rates Q matrix has only one parameter: the forward and
  backswards rates for all characters are all equal.
* **"Sym"**: A symmetrical rates Q matrix forces forward and reverse rates
  to be equal, but allows rates for different characters to differ. It has
  a number of parameters equal to (N^2 - N)/2, where N is the number of
  character states.
* **"ARD"**: An all-rates different Q matrix allows all rates to vary freely.
  It has a number of parameters equal to (N^2 - N).

In this case, we will fit an ARD Q matrix.

We also need to specify how the prior at the root is handled. There are a
few ways to handle weighting the likelihood at the root:

* **"Equal"**: When the likelihood at the root is set to equal, no weighting
  is given to the root being in any particular state. All likelihoods
  for all states are given equal weighting
* **"Equilibrium"**: This setting causes the likelihoods at the root to be
  weighted by the stationary distribution of the Q matrix, as is described
  in Maddison et al 2007.
* **"Fitzjohn"**: This setting causes the likelihoods at the root to be
  weighted by how well each root state would explain the data at the tips,
  as is described in Fitzjohn 2009.

In this case we will use the "Fitzjohn" method.

We can use the ``fitMk`` method with these settings to fit the model. This
function returns a ``dict`` containing the fitted Q matrix, the log-likelihood,
and the weighting at the root node.

.. sourcecode:: ipython

    In [*]: from ivy.chars import mk
    In [*]: mk_results = mk.fit_Mk(tree, charList, Q="ARD", pi="Fitzjohn")
    In [*]: print mk_results["Q"]
    [[-0.01246449  0.01246449]
     [ 0.09898439 -0.09898439]]
    In [*]: print mk_results["Log-likelihood"]
    -11.3009106093
    In [*]: print mk_results["pi"]
    {0: 0.088591248260230959, 1: 0.9114087517397691}

Let's take a look at the results

    In [*]: print mk_results["Q"]
    [[-0.01246449  0.01246449]
     [ 0.09898439 -0.09898439]]

First is the Q matrix. The fitted Q matrix shows the transition rate from i->j,
where i is the row and j is the column. Recall that in this dataset, character
0 corresponds to herbacious and 1 to woody. We can see that the transition
rate from woody to herbacious is higher than the transition from
herbacious to woody.

    In [*]: print mk_results["Log-likelihood"]
    -11.3009106093

Next is the log-likelihood. This is the log-likelihood of the data using
the fitted model

    In [*]: print mk_results["pi"]
    {0: 0.088591248260230959, 1: 0.9114087517397691}

Finally we have pi, the weighting at the root. We can see that there is
higher weighting for the root being in state 1 (woody).

.. TODO: plotting Mk

Bayesian
~~~~~~~~
``ivy`` has a framework in place for using ``pymc`` to sample from a Bayesian
Mk model. The process of fitting a Bayesian Mk model is very similar to fitting
a maximum likelihood model.

The module ``bayesian_models`` has a function ``create_mk_model`` that takes
the same input as ``fitMk`` and creates a ``pymc`` model that can  be sampled
with an MCMC chain

First we create the model.

.. sourcecode:: ipython

    In [*]: from ivy.chars import bayesian_models
    In [*]: from ivy.chars.bayesian_models import create_mk_model
    In [*]: mk_mod = create_mk_model(tree, charList, Qtype="ARD", pi="Fitzjohn")

Now that we have the model, we can use ``pymc`` syntax to set up an MCMC chain.

.. sourcecode:: ipython

    In [*]: import pymc
    In [*]: mk_mcmc = pymc.MCMC(mk_mod)
    In [*]: mk_mcmc.sample(4000, burn=200, thin = 2)
    [-----------------100%-----------------] 2000 of 2000 complete in 4.7 sec

We can access the results using the ``trace`` method of the mcmc object and
giving it the name of the parameter we want. In this case, we want "Qparams"

.. sourcecode:: ipython

    In [*]: mk_mcmc.trace("Qparams")[:]
    array([[ 0.01756608,  0.07222648],
       [ 0.03266443,  0.05712813],
       [ 0.03266443,  0.05712813],
       ...,
       [ 0.01170189,  0.03909211],
       [ 0.01170189,  0.03909211],
       [ 0.00989616,  0.03305975]])

Each element of the trace is an array containing the two fitted Q parameters.
Let's get the 5%, 50%, and 95% percentiles for both parameters

.. sourcecode:: ipython

    In [*]: import numpy as np
    In [*]: Q01 = [ i[0] for i in mk_mcmc.trace("Qparams")[:] ]
    In [*]: Q10 = [ i[1] for i in mk_mcmc.trace("Qparams")[:] ]
    In [*]: np.percentile(Q01, [5,50,95])
    Out[*]: array([ 0.00308076,  0.01844342,  0.06290078])
    In [*]: np.percentile(Q10, [5,50,95])
    Out[*]: array([ 0.03294584,  0.09525662,  0.21803742])

Unsurprisingly, the results are similar to the ones we got from the maximum
likelihood analysis

Hidden-Rates Models
-------------------
``ivy`` has functions for fitting hidden-rates Markov models (HRM) (see Beaulieu et
al. 2013), as well as for performing ancestor-state reconstructions and
visualizations of hidden-rates models.

We will demonstrate these functions using an example dataset, fitting a
two-state character with two different hidden rates or "regimes". Let's
set up our data.


.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: from ivy.chars import hrm
    In [*]: tree = ivy.tree.read("hrm_600tips.newick")
    In [*]: chars = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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

Maximum Likelihood
~~~~~~~~~~~~~~~~~~
``ivy`` can fit a maximum likelihood HRM model using the ``fit_hrm`` function.
Here we will explain the options and uses of this function.

Like the Mk functions, ``ivy``'s HRM functions take as input a tree and a list
of the characters in preorder sequence. These character states are already in
order.

The simplest way to fit an HRM model is to use the ``fit_hrm`` function. Like
the ``fitMk`` function, there are a few possible models we can fit:

* **"Simple"**: Under the simple model, each regime is equivalent to an
  equal-rates Mk model. Transitions between regimes are all constrained to
  be equal. This model fits M+1 parameters where M is the number of regimes.
* **"ARD"**: Under this model, all rates are allowed to vary freely. This
  model fits (N^2 - N)/(2/M) + (M^2-M)*N parameters, where N is the
  number of character states and M is the number of regimes.

Here we will fit a ``Simple`` model.

.. sourcecode:: ipython

    In [*]: fit = hrm.fit_hrm(tree, chars, nregime=2, Qtype="Simple", pi="Equal")
    In [*]: fit
    Out[*]:
    {'Log-likelihood':-204.52351825389133,
    'Q': (array([[-0.04596664,  0.03291299,  0.01305365,  0.        ],
            [ 0.03291299, -0.04596664,  0.        ,  0.01305365],
            [ 0.01305365,  0.        , -0.44655655,  0.4335029 ],
            [ 0.        ,  0.01305365,  0.4335029 , -0.44655655]]),
    'rootLiks': {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})


The output of the ``fit_hrm`` function is a dictionary containing the
fitted Q matrix, the log-likelihood, and the prior weighting at the root (all
values are equal in this case because we set pi to be "Equal")


Now that we have our reconstructed Q matrix, we can perform ancestral state
reconstruction. We will use the ``anc_recon_discrete`` function. This
function can handle both single-regime Mk models and multi-regime
HRM models. This functions takes a tree, the character states at the
tips, and the Q matrix to be used in the reconstruction. We will use
our fitted Q matrix as input. We also provide the prior at the root (in
this case, "Equal"). Because this is a hidden-rates model, we also provide
the number of regimes (in this case, 2)

.. sourcecode:: ipython

    In [*]: from ivy.chars import anc_recon
    In [*]: recon = anc_recon.anc_recon_discrete(tree, chars, fit[0],
                                                 pi="Equal", nregime=2)

The output of this function is an array containing the likelihoods of each
node being in each state. We can use the output of this function to
visualize the reconstructed states on the tree.

.. sourcecode:: ipython

    In [*]: from ivy.interactive import *
    In [*]: from ivy.vis.layers import add_ancrecon_hrm
    In [*]: fig = treefig(tree)
    In [*]: fig.add_layer(add_ancrecon_hrm, recon)


.. image:: _images/hrm_1.png
    :width: 700

Green corresponds to state 0 and blue to state 1. The more saturated colors
correspond to the faster regime and the duller colors to the slower regime.

BAMM-like Mk model
------------------
``ivy`` has code for fitting a `BAMM <http://bamm-project.org/index.html>`_-like
Mk model (also referred to here as a multi-mk model) to a tree, where different
sections of the tree have distinct Mk models associated with them.

We will demonstrate fitting a two-regime BAMM-like Mk model in a Bayesian
context using pymc

.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: from ivy.chars import mk_mr
    In [*]: tree = ivy.tree.read("hrm_600tips.newick")
    In [*]: chars = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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

    In [*]: data = dict(zip([n.label for n in tree.leaves()],chars))

First, we will use the ``ivy`` function ``mk_multi_bayes`` to create the pymc
obejct. We will specify the model using a special array, the ``qidx`` array. The
``qidx`` array provides information for fitting parameters into a Q matrix.

The ``qidx`` for a multi-mk model is a two-dimensional numpy array with four
columns. The first column refers to regime, the second column refers to the row index
of the Q matrix, and the third column refers to the column index of the Q matrix.
The fourth row refers to the parameter identity that will be filled into this
location.

TODO: link to mk_multi_bayes documentation


In this case, we will fit a model where there are two equal-rates mk models
somewhere on the tree, each with two different rates. The ``qidx`` will be
as follows:

.. sourcecode:: ipython

    In [*]: import numpy as np
    In [*]: qidx = np.array([[0,0,1,0],
                             [0,1,0,0],
                             [1,0,1,1],
                             [1,1,0,1]])
    In [*]: my_model = mk_mr.mk_multi_bayes(tree, data,nregime=2,qidx=qidx)

Now we will sample from our model. We will do 100,000 samples with a burn-in
of 10,000 and a thinning factor of 3.

.. sourcecode:: ipython

    In [*] my_model.sample(100000,burn=10000,thin=3)

Now we can look at the output. Seeing the fitted parameters is easy.

.. sourcecode:: ipython

    In [*]: print(np.mean(my_model.trace("Qparam_0")[:])) # Q param 0
    In [*]: print(np.mean(my_model.trace("Qparam_1")[:])) # Q param 1

Seeing the location of the fitted switchpoints is a little trickier. We can
use the plotting layer ``add_tree_heatmap`` to see where the switchpoint
was reconstructed at.

.. sourcecode:: ipython

    In [*]: from ivy.interactive import *
    In [*]: fig = treefig(tree)
    In [*]: switchpoint = my_model.trace("switch_0")[:]
    In [*]: fig.add_layer(ivy.vis.layers.add_tree_heatmap,switchpoint)

.. image:: _images/mkmr_1.png
    :width: 700
