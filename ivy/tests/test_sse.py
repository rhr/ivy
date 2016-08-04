"""
Unittests for SSE methods
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
import numpy as np
from ivy.chars.sse import sse


class sse_methods(unittest.TestCase):
    def setUp(self):
        self.threetiptree = ivy.tree.read(u'((A:1,B:1):1,C:2)root;')

        self.threetipdata = {"A":1, "B":1,"C":0}

    def test_bisse_threetip_unequalparams(self):
        root = self.threetiptree
        data = self.threetipdata

        params = np.array([0.1, 0.2, 0.01, 0.02, 0.05, 0.07])

        calc_lik = sse.bisse_odeiv(root,data,params,condition_on_surv=True)
        true_lik = -5.007626
        self.assertTrue(np.isclose(calc_lik,true_lik,atol=1e-7))
    def test_classe2state_threetip_twostate(self):
        root = self.threetiptree
        data = self.threetipdata
        params = {'lambda000':0.3,'lambda001':0.1,'lambda011':0.01,'lambda100':0.01,'lambda101':0.1,'lambda111':0.2,'mu0':0.01,'mu1':0.01,'q01':0.2,'q10':0.1}

        calc_lik = sse.classe_odeiv_2state(root,data,params,condition_on_surv=True)
        true_lik = -4.912378
        print(calc_lik)
        self.assertTrue(np.isclose(calc_lik,true_lik,atol=1e-2))
    def test_classe_threetip_twostate(self):
        root = self.threetiptree
        data = self.threetipdata
        lambdaparams = np.array([[[0.3,0.1  ],
                                  [0  , 0.01]],

                                 [[0.01,0.1],
                                  [0   ,0.2]]])

if __name__=="__main__":
    unittest.main()
