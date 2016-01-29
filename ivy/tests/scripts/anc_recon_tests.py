import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.mk import _create_nodelist
import math
from ivy.chars.expokit import cyexpokit

tree = ivy.tree.read("(((A:1,B:1)E:1,C:2)F:1,D:3)R;")
chars = [1,1,0,0]
Q = np.array([[-0.1,0.1],[0.1,-0.1]])
p = None
pi = "Fitzjohn"
preallocated_arrays = None
