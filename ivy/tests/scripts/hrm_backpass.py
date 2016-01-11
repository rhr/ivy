import ivy
from ivy.chars import discrete
import numpy as np
from ivy.chars import bayesian_models
import pymc
import matplotlib.pyplot as plt
from ivy.chars.expokit import cyexpokit
import math

tree = ivy.tree.read("((1:1,(2:0.8606963753,(((3:0.05717681656,4:0.05717681656):0.1249697997,5:0.1821466162):0.4326289449,((6:0.02675514074,7:0.02675514074):0.3210121364,(8:0.2582442539,(9:0.0832722267,10:0.0832722267):0.1749720272):0.08952302326):0.2670082839):0.2459208142):0.1393036247):1,((((11:0.04422729573,12:0.04422729573):0.4961649323,(13:0.3446901329,14:0.3446901329):0.1957020951):0.2248653717,(((15:0.009354454367,16:0.009354454367):0.06248862166,(17:0.02880343044,18:0.02880343044):0.04303964559):0.148588294,19:0.22043137):0.5448262298):0.2347424003,20:1):1):0;")
chars = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1]

Q = ivy.chars.discrete.fill_Q_matrix(2,2, [0.0,1.0,0.00,1.0,1.0,5.5,1.0,5.5])

nregime = 2
p = None
pi = "Fitzjohn"
returnPi = False
preallocated_arrays = None
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
#ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2)
temp2 = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states = temp[0])
temp3 = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states = temp2[0])



liks = np.zeros(50)

tmp = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2)
for i in range(50):
    tmp = ivy.chars.hrm.hrm_back_mk(tree, chars, Q, 2, tip_states=tmp[0])
    liks[i] = tmp[1]
