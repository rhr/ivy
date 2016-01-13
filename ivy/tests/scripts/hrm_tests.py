import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.hrm import _create_hrmnodelist
from ivy.chars.hrm import *

tree = ivy.tree.read("support/hrm_300tips.newick")
chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
        1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0]

nregime = 2
# out = bayesian_models.hrm_bayesian(tree, chars, Qtype="Simple", nregime=2)
#
# outMC = pymc.MCMC(out)
# outMC.sample(10000, 400, 3)

# plt.plot(outMC.trace("parA")[:])
# plt.plot(outMC.trace("parB")[:])
# plt.plot(outMC.trace("parC")[:])
#
# for i in ["parA", "parB", "parC"]:
#     print np.percentile(outMC.trace(i)[:], [2.5, 50, 97.5])
# # True values are 0.01, 0.1, and 0.6
Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])

hrm_back_mk(tree, chars, Q, nregime)



out = bayesian_models.hrm_multipass_bayesian(tree, chars, Qtype="Simple", nregime=2)
outMC = pymc.MCMC(out)

import cProfile
# cProfile.run('outMC.sample(50,0,1)')
outMC.sample(10000,400,3)
#
#

plt.plot(outMC.trace("wr-par0")[:])
plt.plot(outMC.trace("br-par")[:])
plt.plot(outMC.trace("wr-par1")[:])

np.median(outMC.trace("wr-par0")[:])
np.median(outMC.trace("br-par")[:])
np.median(outMC.trace("wr-par1")[:])


out2 = bayesian_models.hrm_bayesian(tree, chars, Qtype="Simple", nregime=2)

out2MC = pymc.MCMC(out2)
#cProfile.run('out2MC.sample(50,0,1)')

Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])

#cProfile.run("discrete.hrm_mk(tree, chars, Q, 2)")
#cProfile.run("discrete.hrm_back_mk(tree, chars, Q, 2)")

#cProfile.run("discrete.hrm_multipass(tree, chars, Q, 2)")

nullval = np.inf
nchar = len(set(chars)) * nregime
nt =  len(tree.descendants())
charlist = range(nchar)
nobschar = len(set(chars))
Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])
t_Q = Q.copy()
# Empty p matrix
p = np.empty([nt, nchar, nchar], dtype = np.double, order="C")
# Empty likelihood array
nodelist,t,childlist = _create_hrmnodelist(tree, chars, nregime)
nodelistOrig = nodelist.copy() # Second copy to refer back to
# Empty root prior array
rootpriors = np.empty([nchar], dtype=np.double)

# Upper bounds
treelen = sum([ n.length for n in tree.leaves()[0].rootpath() if n.length]+[
               tree.leaves()[0].length])
upperbound = len(tree.leaves())/treelen
# Giving internal function access to these arrays.
   # Warning: can be tricky
   # Need to make sure old values
   # Aren't accidentally re-used

var = {"Q": Q, "p": p, "t":t, "nodelist":nodelist, "charlist":charlist,
       "nodelistOrig":nodelistOrig, "upperbound":upperbound,
       "root_priors":rootpriors, "nullval":nullval, "t_Q":t_Q,
       "p_up":p.copy(), "v":np.zeros([nchar]),"tmp":np.zeros([nchar+1]),
       "motherRow":np.zeros([nchar+1]),"childlist":childlist}

var["nodelist-up"] =var["nodelist"].copy()

tip_states = None

pi = "Fitzjohn"

np.copyto(var["nodelist"], var["nodelistOrig"])
var["root_priors"].fill(1.0)
import copy
var1 = copy.deepcopy(var)

preallocated_arrays = var

cProfile.run("hrm_multipass(tree, chars, Q, 2, preallocated_arrays=var)")




Qtype = "Simple"
Qparams = np.array([0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])


pi = "Fitzjohn"
l_single = discrete.create_likelihood_function_hrm_mk(tree=tree, chars=chars,
        nregime=nregime, Qtype="ARD", pi=pi, min=False)

def loop1():
    for i in range(5):
        l_single(np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]))


cProfile.run("loop1()")

l_multipass = discrete.create_likelihood_function_hrmmultipass_mk(tree=tree, chars=chars,
        nregime=nregime, Qtype="ARD", pi=pi, min=False)
def loop2():
    for i in range(5):
        l_multipass(Qparams)

cProfile.run("loop2()")


var = copy.deepcopy(var1)

def loop3():
    for i in range(5*11):
        discrete.hrm_back_mk(tree, chars, Q, nregime, preallocated_arrays=var)

cProfile.run("loop3()")

var = copy.deepcopy(var1)

def loop4(var1):
    for i in range(5):
        var = copy.deepcopy(var1)
        discrete.hrm_multipass(tree, chars, Q, nregime, preallocated_arrays=var)


cProfile.run("loop4(var1)")


var = copy.deepcopy(var1)
def loop5():
    discrete.hrm_back_mk(tree, chars, Q, nregime, preallocated_arrays=var)
cProfile.run("loop5()")


var = copy.deepcopy(var1)
def loop6():
    discrete.hrm_mk(tree, chars, Q, nregime, preallocated_arrays=var)
cProfile.run("loop6()")

var = copy.deepcopy(var1)
cProfile.run("discrete.hrm_back_mk(tree, chars, Q, nregime, preallocated_arrays=var)")



tip_states=None
preallocated_arrays=var
p=None
returnPi=False


















cProfile.run("l(Qparams)")
cProfile.run("l_single(Qparams)")
