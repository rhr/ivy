import ivy
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
from scipy import cluster

from ivy.vis import layers
from ivy.sim.discrete_sim import sim_discrete
from ivy.chars import mk, hrm, recon

from ivy.interactive import *

tree = ivy.tree.read("support/hrm_600tips.newick")


# Generating model
gen_Q = np.array([[-0.01, 0.01],
               [0.01, -0.01]])
# Creating the simulation
np.random.seed(4)
sim_tree = sim_discrete(tree, gen_Q)

chars = [ n.sim_char["sim_state"] for n in sim_tree.leaves()]


##########################
# 8-param MLE
##########################
hrm_mle = hrm.fit_hrm(tree, chars, nregime=2, Qtype="ARD")
Q = hrm_mle[0]
mask=np.ones([4,4],dtype=bool)
mask[np.diag_indices(4)] = False
mask[0,3]=False;mask[1,2]=False;mask[2,1]=False;mask[3,0]=False
Q_params = Q[mask]


#########################
# Clustering
#########################
Q_dist = np.array(zip(list(Q_params), [0]*8))
scipy.cluster.hierarchy.fclusterdata(Q_dist, 0.01)


ts = np.logspace(-10, 0, 11)
candidate_models = set([tuple(scipy.cluster.hierarchy.fclusterdata(Q_dist, i)) for i in ts])

def extract_Qparams(Q, nregime):
    nchar = Q.shape[0]
    nobschar = nchar/nregime

    n_wr = (nobschar**2-nobschar)
    n_br = (nregime**2-nregime)*nobschar

    Q_params = np.zeros([n_wr*nregime + n_br])

    for i in range(nregime):
        subQ = slice(i*nobschar,(i+1)*nobschar)
        mask = np.ones([nobschar,nobschar], dtype=bool)
        mask[np.diag_indices(nobschar)]=False
        np.copyto(Q_params[i*n_wr:(i+1)*n_wr], Q[subQ,subQ][mask])

    combs = list(itertools.combinations(range(nregime),2))
    revcombs = [tuple(reversed(i)) for i in combs]
    submatrix_indices = [x for s in [[combs[i]] + [revcombs[i]] for i in range(len(combs))] for x in s]
    for i,submatrix_index in enumerate(submatrix_indices):
        my_slice0 = slice(submatrix_index[0]*nobschar, (submatrix_index[0]+1)*nobschar)
        my_slice1 = slice(submatrix_index[1]*nobschar, (submatrix_index[1]+1)*nobschar)
        nregimeswitch =(nregime**2 - nregime)*2
        Q_params[n_wr*nregime + i*nobschar:n_wr*nregime+(i+1)*nobschar] = Q[my_slice0,my_slice1][np.diag_indices(nobschar)]
