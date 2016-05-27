import ivy
import numpy as np
import matplotlib.pyplot as plt
import random
import pickle
import pandas

from ivy.interactive import *
from ivy.vis import layers
from ivy.sim import discrete_sim
from colour import Color
from matplotlib.colors import LinearSegmentedColormap
np.set_printoptions(suppress=True, precision=5)

tree = ivy.tree.read("/home/cziegler/Dropbox/multiregime-discrete/datasets/mammal_diet/MammalPhyloInDataset.newick")
data = pandas.read_csv("/home/cziegler/Dropbox/multiregime-discrete/datasets/mammal_diet/MammalDietOnTree.csv")

Q = np.array([[ -0.00897,   0.0087 ,   0.00027,   0.     ,   0.     ,   0.     ],
       [  0.11584,  -0.15307,   0.     ,   0.     ,   0.03723,   0.     ],
       [  0.     ,   0.11584, -39.10997,   0.     ,   0.     ,  38.99413],
       [  0.     ,   0.     ,   0.     ,  -2.64026,   2.64026,   0.     ],
       [  0.     ,   0.01447,   0.     ,   0.1156 ,  -0.1377 ,   0.00763],
       [  0.     ,   0.     ,   0.     ,   0.     ,   0.00696,  -0.00696]])

chars = dict(zip(data.Binomial, data.DietCode.astype(int)))


fig = treefig(tree)
treeplot = fig.tree

liks = ivy.chars.recon.anc_recon_cat(tree, chars, Q=Q, nregime=2)


def color_blender_1(value, start, end):
    """
    Smooth transition between two values

    value (float): Percentage along color map
    start: starting value of color map
    end: ending value of color map
    """
    return start + (end-start)*value

def color_map(value, col1, col2):
    """
    Return RGB for value based on minimum and maximum colors
    """
    r = color_blender_1(value, col1[0], col2[0])
    g = color_blender_1(value, col1[1], col2[1])
    b = color_blender_1(value, col1[2], col2[2])

    return(r,g,b)
