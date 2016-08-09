"""
Unittests for SSE methods
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
import numpy as np
from ivy.chars.sse import sse
from ivy.chars import discrete

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
    def test_classe_threetip_twostate(self):
        root = self.threetiptree
        data = self.threetipdata
        lambdaparams = np.array([[[0.3,0.1  ],
                                  [0  , 0.01]],

                                 [[0.01,0.1],
                                  [0   ,0.2]]])
        params = np.array([2,0.3,0.1,0.01,0.01,0.1,0.2,0.01,0.01,0.2,0.1])
        calc_lik_ncos = sse.classe_likelihood(root,data,2,params,False)
        true_lik_ncos = -5.962547

        calc_lik_cos = sse.classe_likelihood(root,data,2,params,True)
        true_lik_cos = -4.912378
        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1e-1))
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1e-1))

    def test_classe_threetip_threestate(self):
        root = self.threetiptree
        data = self.threetipdata

        params = np.array([3,0.3,0.1,0.7,0.01,0.8,0.05,0.01,0.1,0.9,0.2,0.004,0.3,0.3,0.11,0.14,0.15,0.17,0.19,0.01,0.01,0.1,0.2,0.4,0.1,0.01,0.7,0.6])
        calc_lik_ncos = sse.classe_likelihood(root,data,3,params,False)
        print(calc_lik_ncos)
        true_lik_ncos = -9.6298

        calc_lik_cos = sse.classe_likelihood(root,data,3,params,True)
        print(calc_lik_cos)
        true_lik_cos = -10.00689
        self.assertTrue(np.isclose(calc_lik_ncos,true_lik_ncos,atol=1))
        self.assertTrue(np.isclose(calc_lik_cos,true_lik_cos,atol=1))
class sse_calculations(unittest.TestCase):
    def setUp(self):
        self.params2state = ["lambda000","lambda001","lambda011","lambda100","lambda101","lambda111","mu0","mu1","q01","q10"]
        self.params3state = ["lambda000","lambda001","lambda002","lambda011","lambda012","lambda022",
                             "lambda100","lambda101","lambda102","lambda111","lambda112","lambda122",
                             "lambda200","lambda201","lambda202","lambda211","lambda212","lambda222",
                             "mu0","mu1","mu2","q01","q02","q10","q12","q20","q21"]
    def test_unpackparams2state_lambda(self):
        lambda000 = discrete.sse_get_lambda(self.params2state,0,0,0,2)
        lambda001 = discrete.sse_get_lambda(self.params2state,0,0,1,2)
        lambda010 = discrete.sse_get_lambda(self.params2state,0,1,0,2)
        lambda011 = discrete.sse_get_lambda(self.params2state,0,1,1,2)
        lambda100 = discrete.sse_get_lambda(self.params2state,1,0,0,2)
        lambda101 = discrete.sse_get_lambda(self.params2state,1,0,1,2)
        lambda110 = discrete.sse_get_lambda(self.params2state,1,1,0,2)
        lambda111 = discrete.sse_get_lambda(self.params2state,1,1,1,2)

        self.assertEquals(lambda000,"lambda000")
        self.assertEquals(lambda001,"lambda001")
        self.assertEquals(lambda010,0)
        self.assertEquals(lambda100,"lambda100")
        self.assertEquals(lambda101,"lambda101")
        self.assertEquals(lambda110,0)
        self.assertEquals(lambda111,"lambda111")

    def test_unpackparams2state_mu(self):
        mu0 = discrete.sse_get_mu(self.params2state,0,2)
        mu1 = discrete.sse_get_mu(self.params2state,1,2)

        self.assertEquals(mu0,"mu0")
        self.assertEquals(mu1,"mu1")
    def test_unpackparams2state_q(self):
        q00 = discrete.sse_get_qij(self.params2state,0,0,2)
        q01 = discrete.sse_get_qij(self.params2state,0,1,2)
        q10 = discrete.sse_get_qij(self.params2state,1,0,2)
        q11 = discrete.sse_get_qij(self.params2state,1,1,2)

        self.assertEquals(q00,0)
        self.assertEquals(q01,"q01")
        self.assertEquals(q10,"q10")
        self.assertEquals(q11,0)
    def test_unpackparams3state_lambda(self):
        lambda000 = discrete.sse_get_lambda(self.params3state,0,0,0,3)
        lambda001 = discrete.sse_get_lambda(self.params3state,0,0,1,3)
        lambda002 = discrete.sse_get_lambda(self.params3state,0,0,2,3)
        lambda010 = discrete.sse_get_lambda(self.params3state,0,1,0,3)
        lambda011 = discrete.sse_get_lambda(self.params3state,0,1,1,3)
        lambda012 = discrete.sse_get_lambda(self.params3state,0,1,2,3)
        lambda020 = discrete.sse_get_lambda(self.params3state,0,2,0,3)
        lambda021 = discrete.sse_get_lambda(self.params3state,0,2,1,3)
        lambda022 = discrete.sse_get_lambda(self.params3state,0,2,2,3)
        lambda100 = discrete.sse_get_lambda(self.params3state,1,0,0,3)
        lambda101 = discrete.sse_get_lambda(self.params3state,1,0,1,3)
        lambda102 = discrete.sse_get_lambda(self.params3state,1,0,2,3)
        lambda110 = discrete.sse_get_lambda(self.params3state,1,1,0,3)
        lambda111 = discrete.sse_get_lambda(self.params3state,1,1,1,3)
        lambda112 = discrete.sse_get_lambda(self.params3state,1,1,2,3)
        lambda120 = discrete.sse_get_lambda(self.params3state,1,2,0,3)
        lambda121 = discrete.sse_get_lambda(self.params3state,1,2,1,3)
        lambda122 = discrete.sse_get_lambda(self.params3state,1,2,2,3)
        lambda200 = discrete.sse_get_lambda(self.params3state,2,0,0,3)
        lambda201 = discrete.sse_get_lambda(self.params3state,2,0,1,3)
        lambda202 = discrete.sse_get_lambda(self.params3state,2,0,2,3)
        lambda210 = discrete.sse_get_lambda(self.params3state,2,1,0,3)
        lambda211 = discrete.sse_get_lambda(self.params3state,2,1,1,3)
        lambda212 = discrete.sse_get_lambda(self.params3state,2,1,2,3)
        lambda220 = discrete.sse_get_lambda(self.params3state,2,2,0,3)
        lambda221 = discrete.sse_get_lambda(self.params3state,2,2,1,3)
        lambda222 = discrete.sse_get_lambda(self.params3state,2,2,2,3)

        self.assertEquals(lambda000,"lambda000")
        self.assertEquals(lambda001,"lambda001")
        self.assertEquals(lambda002,"lambda002")
        self.assertEquals(lambda010,0)
        self.assertEquals(lambda011,"lambda011")
        self.assertEquals(lambda012,"lambda012")
        self.assertEquals(lambda020,0)
        self.assertEquals(lambda021,0)
        self.assertEquals(lambda022,"lambda022")

        self.assertEquals(lambda100,"lambda100")
        self.assertEquals(lambda101,"lambda101")
        self.assertEquals(lambda102,"lambda102")
        self.assertEquals(lambda110,0)
        self.assertEquals(lambda111,"lambda111")
        self.assertEquals(lambda112,"lambda112")
        self.assertEquals(lambda120,0)
        self.assertEquals(lambda121,0)
        self.assertEquals(lambda122,"lambda122")

        self.assertEquals(lambda200,"lambda200")
        self.assertEquals(lambda201,"lambda201")
        self.assertEquals(lambda202,"lambda202")
        self.assertEquals(lambda210,0)
        self.assertEquals(lambda211,"lambda211")
        self.assertEquals(lambda212,"lambda212")
        self.assertEquals(lambda220,0)
        self.assertEquals(lambda221,0)
        self.assertEquals(lambda222,"lambda222")
    def test_unpackparams3state_mu(self):
        mu0 = discrete.sse_get_mu(self.params3state,0,3)
        mu1 = discrete.sse_get_mu(self.params3state,1,3)
        mu2 = discrete.sse_get_mu(self.params3state,2,3)

        self.assertEquals(mu0,"mu0")
        self.assertEquals(mu1,"mu1")
        self.assertEquals(mu2,"mu2")
    def test_unpackparams3state_q(self):
        q00 = discrete.sse_get_qij(self.params3state,0,0,3)
        q01 = discrete.sse_get_qij(self.params3state,0,1,3)
        q02 = discrete.sse_get_qij(self.params3state,0,2,3)

        q10 = discrete.sse_get_qij(self.params3state,1,0,3)
        q11 = discrete.sse_get_qij(self.params3state,1,1,3)
        q12 = discrete.sse_get_qij(self.params3state,1,2,3)

        q20 = discrete.sse_get_qij(self.params3state,2,0,3)
        q21 = discrete.sse_get_qij(self.params3state,2,1,3)
        q22 = discrete.sse_get_qij(self.params3state,2,2,3)


        self.assertEquals(q00,0)
        self.assertEquals(q01,"q01")
        self.assertEquals(q02,"q02")
        self.assertEquals(q10,"q10")
        self.assertEquals(q11,0)
        self.assertEquals(q12,"q12")
        self.assertEquals(q20,"q20")
        self.assertEquals(q21,"q21")
        self.assertEquals(q22,0)



if __name__=="__main__":
    unittest.main()
