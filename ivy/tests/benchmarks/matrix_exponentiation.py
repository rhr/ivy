import unittest
import ivy
from ivy.chars.expokit import cyexpokit
import numpy as np
import random





Q = np.array([[-1,1,0,0],
                      [0,-1,1,0],
                      [0,0,-1,1],
                      [0,0,0,0]], dtype=np.double, order='C')
                      
                      
t = np.array([ random.random() for x in range(10000) ])


%timeit -n2 -r3 cyexpokit.dexpm_tree(Q, t)
