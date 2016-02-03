import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.mk import _create_nodelist
import math
from ivy.chars.expokit import cyexpokit
from ivy.chars.anc_recon import anc_recon, anc_recon_purepy
np.set_printoptions(precision=5, suppress=True)

tree = ivy.tree.read("(((A:1,B:1)E:1,C:2)F:1,D:3)R;")
chars = [1,1,0,0]
Q = np.array([[-0.1,0.1],[0.1,-0.1]])
p = None
pi = "Fitzjohn"
ars = None

#####################

ars = ivy.chars.anc_recon.create_ancrecon_ars(tree, chars)
p = np.empty([len(ars["t"]), Q.shape[0],
             Q.shape[1]], dtype = np.double, order="C")


%timeit anc_recon(tree, chars, Q, ars=ars, p=p)
%timeit anc_recon_purepy(tree, chars, Q, ars=ars, p=p)
