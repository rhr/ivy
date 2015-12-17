import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.expokit import cyexpokit

tree = ivy.tree.read("support/randtree10tips.newick")
chars = [0,0,0,1,1,1,0,0,0,1]

Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.05,0.01,0.03,0.005,0.15,0.3,0.01,0.2])

nregime = 2
p = None
pi = "Fitzjohn"
returnPi = False
preallocated_arrays = None
