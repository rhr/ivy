import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.mk import _create_nodelist
import math
from ivy.chars.expokit import cyexpokit

tree = ivy.tree.read("((A:1,B:1)F:2,((C:1,D:1)G:1,E:2)H:1)R;")
chars = [0,0,1,1,0]
Q = np.array([[-0.1,0.1],[0.1,-0.1]])
p = None
pi = "Fitzjohn"
preallocated_arrays = None
