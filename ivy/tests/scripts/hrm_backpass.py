import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.expokit import cyexpokit
import math
from ivy.chars.hrm import *
from ivy.chars.hrm import _create_hrmnodelist

tree = ivy.tree.read("((1:1,(2:0.8606963753,(((3:0.05717681656,4:0.05717681656):0.1249697997,5:0.1821466162):0.4326289449,((6:0.02675514074,7:0.02675514074):0.3210121364,(8:0.2582442539,(9:0.0832722267,10:0.0832722267):0.1749720272):0.08952302326):0.2670082839):0.2459208142):0.1393036247):1,((((11:0.04422729573,12:0.04422729573):0.4961649323,(13:0.3446901329,14:0.3446901329):0.1957020951):0.2248653717,(((15:0.009354454367,16:0.009354454367):0.06248862166,(17:0.02880343044,18:0.02880343044):0.04303964559):0.148588294,19:0.22043137):0.5448262298):0.2347424003,20:1):1):0;")
chars = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]

Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])

nregime = 2
p = None
pi = "Fitzjohn"
returnPi = False
preallocated_arrays = None
tip_states = None
Qtype="ARD"
min=True
Qparams = np.array([0.0,1.0,0.0,1.0,1.0,5.5,1.0,5.5])




f = discrete.create_likelihood_function_hrmmultipass_mk(tree, chars, 2, "ARD")

Qparams = np.array([0.0,1.0,0.0,1.0,1.0,5.5,1.0,5.5])





#
# tree.ape_node_idx()
#
# for t in tree:
#     t.label = t.apeidx
#
#
# treeni = 38
#
#
# node1ni = 18
# node1n = preallocated_arrays["nodelist-up"][18]

#
# node2ni = 17
# node2n = preallocated_arrays["nodelist-up"][17]
#
# for i in preallocated_arrays["nodelist"]:
#     print i[:-1]/sum(i[:-1])
# for t in tips:
#     print t[:-1]/np.sum(t[:-1])


temp = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2)

temp2 = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states = temp[0])
temp3 = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states = temp2[0])



liks = np.zeros(50)

tmp = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2)
for i in range(50):
    tmp = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states=tmp[0])
    liks[i] = tmp[1]



def hrm_backpass_test(tree, chars, Q, nrep=50):
    nobschar = len(set(chars))
    nregime = Q.shape[0]/nobschar

    liks = np.zeros(nrep)
    tmps = np.empty(nrep+1, dtype=object)
    tmps[0] = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2)
    for i in range(nrep):
        tmps[i+1] = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, nregime, tip_states=tmp[0])
        liks[i] = tmps[i+1][1]
    return liks, tmps


mr_tree = ivy.tree.read("support/Mk_two_regime_tree.newick")
mr_chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1,
            0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0]

Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.1,0.1,0.1,0.1,0.1,0.8,0.1,0.8])
Q2 = _random_Q_matrix(2,2)

l,t = hrm_backpass_test(mr_tree, mr_chars, Q)
l2,t2 = hrm_backpass_test(mr_tree, mr_chars, Q2)


fastNodes = np.array([ n.pi for n in mr_tree.mrca(mr_tree.grep("f")).preiter()])
slowNodes = np.array([n.pi for n in mr_tree.descendants()if not n.pi in fastNodes])

postLeaves = [ tr.pi for tr in mr_tree if tr.isleaf ]

for i,r in enumerate(t[0]):
    if postLeaves[i] in fastNodes:
        reg = "F"
    else:
        reg = "S"
    n = [tr for tr in mr_tree if tr.pi == postLeaves[i]][0].li

    print r[:-1]/np.sum(r[:-1]), reg, mr_chars[n]


count = 0
for i,n in enumerate(l_2[1]):
    tmp = n[:-1]/sum(n[:-1])
    if true_chars[i] in [0,2]:
        non = [1,3]
    else:
        non = [0,2]
    n[non] = 0.0
    tmp = n[:-1]/sum(n[:-1])

    print tmp, tmp.argmax(),  true_chars[i]
    if tmp.argmax() != true_chars[i]:
        count+=1
