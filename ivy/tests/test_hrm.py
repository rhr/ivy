"""
Unittests for HRM functions
"""
import unittest
import ivy
from ivy.chars import hrm
import numpy as np
import math
import scipy
from ivy.chars.expokit import cyexpokit


class hrmMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1, 0.1], [0.2, -0.2]])
        self.Q3x3_sym = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])
        self.randQ = np.array([[-2,1,1],[1,-2,1],[1,1,-2]], dtype=np.double)

        self.charstates_011 = [0,1,1]
        self.charstates_01 = [0,1]
        self.randChars5 = [1,2,2,1,0]
        self.randChars10 = [0,2,1,1,1,0,0,1,2,2]
        self.randChars600 = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                        0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1,
                        1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                        1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1,
                        1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]



        self.threetiptree = ivy.tree.read("((A:1,B:1)C:1,D:2)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")
        self.randTree5 = ivy.tree.read("support/randtree5tips.newick")
        self.randTree10 = ivy.tree.read("support/randtree10tips.newick")
        self.randTree600 = ivy.tree.read("support/hrm_600tips.newick")
    def test_hrmMk_threetiptree_matchesByHand(self):
        """
        Two observed states: 0 and 1
        Two hidden states per observed state: fast and slow
        """
        tree = self.threetiptree
        chars = [0,1,1]

        # Qarray rows: 0S, 1S, 0F, 1F
        # State transitions more likely than rate transitions
        Q = np.array([[-.15, .1, 0.05, 0],[0.05,-.12,0,0.07],[0.06,0,-.26, .2],[0,0.08,0.3,-.38]])
        t = np.array([i.length for i in tree.descendants()])
        # Tips are assumed to be in both hidden states at once
        # Likelihoods for tip A
        L0SA = 1;L0FA = 1;L1SA = 0;L1FA = 0
        # Likelhoods for tip B
        L0SB = 0;L0FB = 0;L1SB = 1;L1FB = 1
        # Likelihoods for tip D
        L0SD = 0;L0FD = 0;L1SD = 1;L1FD = 1

        p = cyexpokit.dexpm_tree(Q, t)

        pvals = {}
        for i,node in enumerate(["C","A","B","D"]):
            for i1,state1 in enumerate(["0S","1S","0F","1F"]):
                for i2,state2 in enumerate(["0S","1S","0F","1F"]):
                    pvals[state1 + state2 + node] = p[i,i1,i2]

        L0SC = (pvals["0S0SA"] * L0SA + pvals["0S1SA"] * L1SA + pvals["0S0FA"] * L0FA + pvals["0S1FA"] * L1FA) *\
               (pvals["0S0SB"] * L0SB + pvals["0S1SB"] * L1SB + pvals["0S0FB"] * L0FB + pvals["0S1FB"] * L1FB)
        L0FC = (pvals["0F0SA"] * L0SA + pvals["0F1SA"] * L1SA + pvals["0F0FA"] * L0FA + pvals["0F1FA"] * L1FA) *\
               (pvals["0F0SB"] * L0SB + pvals["0F1SB"] * L1SB + pvals["0F0FB"] * L0FB + pvals["0F1FB"] * L1FB)
        L1SC = (pvals["1S0SA"] * L0SA + pvals["1S1SA"] * L1SA + pvals["1S0FA"] * L0FA + pvals["1S1FA"] * L1FA) *\
               (pvals["1S0SB"] * L0SB + pvals["1S1SB"] * L1SB + pvals["1S0FB"] * L0FB + pvals["1S1FB"] * L1FB)
        L1FC = (pvals["1F0SA"] * L0SA + pvals["1F1SA"] * L1SA + pvals["1F0FA"] * L0FA + pvals["1F1FA"] * L1FA) *\
               (pvals["1F0SB"] * L0SB + pvals["1F1SB"] * L1SB + pvals["1F0FB"] * L0FB + pvals["1F1FB"] * L1FB)

        L0Sr = (pvals["0S0SC"] * L0SC + pvals["0S1SC"] * L1SC + pvals["0S0FC"] * L0FC + pvals["0S1FC"] * L1FC) *\
               (pvals["0S0SD"] * L0SD + pvals["0S1SD"] * L1SD + pvals["0S0FD"] * L0FD + pvals["0S1FD"] * L1FD)
        L0Fr = (pvals["0F0SC"] * L0SC + pvals["0F1SC"] * L1SC + pvals["0F0FC"] * L0FC + pvals["0F1FC"] * L1FC) *\
               (pvals["0F0SD"] * L0SD + pvals["0F1SD"] * L1SD + pvals["0F0FD"] * L0FD + pvals["0F1FD"] * L1FD)
        L1Sr = (pvals["1S0SC"] * L0SC + pvals["1S1SC"] * L1SC + pvals["1S0FC"] * L0FC + pvals["1S1FC"] * L1FC) *\
               (pvals["1S0SD"] * L0SD + pvals["1S1SD"] * L1SD + pvals["1S0FD"] * L0FD + pvals["1S1FD"] * L1FD)
        L1Fr = (pvals["1F0SC"] * L0SC + pvals["1F1SC"] * L1SC + pvals["1F0FC"] * L0FC + pvals["1F1FC"] * L1FC) *\
               (pvals["1F0SD"] * L0SD + pvals["1F1SD"] * L1SD + pvals["1F0FD"] * L0FD + pvals["1F1FD"] * L1FD)

        predictedLikelihood = math.log(L0Sr*.25 + L0Fr*.25 + L1Sr * .25 + L1Fr *.25)
        corHMMLikelihood = -2.980018

        calculatedLikelihood = hrm.hrm_mk(tree, chars, Q,2, pi="Equal")

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    def test_hrmMk_twocharsthreeregime_matchescorHMM(self):
        tree = self.randTree10
        chars = [0,0,0,1,1,1,0,0,0,1]

        Q = np.array([[-0.60144712,  0.43291497,  0.16853215,  0.        ,  0.        ,
                             0.        ],
                           [ 0.06749584, -0.28697994,  0.        ,  0.2194841 ,  0.        ,
                             0.        ],
                           [ 0.87295237,  0.        , -2.99021064,  0.80831725,  1.30894102,
                             0.        ],
                           [ 0.        ,  0.70681107,  0.91210804, -2.91883608,  0.        ,
                             1.29991697],
                           [ 0.        ,  0.        ,  1.8858193 ,  0.        , -3.45502732,
                             1.56920802],
                           [ 0.        ,  0.        ,  0.        ,  1.67920079,  1.6287939 ,
                            -3.3079947 ]])


        corHMMLik = -12.53562
        calculatedLikelihood = hrm.hrm_mk(tree, chars, Q, 3, pi="Equal")

        self.assertTrue(np.isclose(corHMMLik, calculatedLikelihood))
    def test_hrmMk_600tiptree_matchescorHMM(self):
        tree = self.randTree600
        Q = np.array([[-.06, .05, .01, 0],
               [.01, -.02, 0, .01],
               [.01, 0, -.71, .7],
                [0, .01, .5, -.51]])
        chars = self.randChars600

        corHMMLikelihood = -203.0632

        self.assertTrue(np.isclose(hrm.hrm_mk(tree, chars, Q, 2, pi="Equal"),
                                    corHMMLikelihood))

    def test_createHrmMkLikelihood_simpleLikelihood_createsproperfunction(self):
        tree = self.threetiptree
        chars = [0,1,1]
        Q = np.array([[-0.2,  0.1,  0.1,  0. ],
                   [ 0.1, -0.2,  0. ,  0.1],
                   [ 0.1,  0. , -0.2,  0.1],
                   [ 0. ,  0.1,  0.1, -0.2]])
        Qparams = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

        f = hrm.create_likelihood_function_hrm_mk(tree, chars, 2, "ARD", pi="Equal")
        val = hrm.hrm_mk(tree, chars, Q, 2, pi="Equal")

        self.assertTrue(np.isclose(f(Qparams), -1*val))

    def test_fitMkARD_600tiptree_matchescorHMM(self):
        tree = self.randTree600
        chars = self.randChars600

        corHMMQ = np.array([[ -0.2028,   0.2028,   0.    ,   0.    ],
                            [  0.0202,  -0.0202,   0.    ,   0.    ],
                            [  0.0412,   0.    ,  -0.1842,   0.1431],
                            [  0.    ,  14.2146,  65.3782, -79.5928]])

        out = hrm.fit_hrm(tree, chars, 2,Qtype="ARD",startingvals=[0.05]*8)

        ivyQ = out["Q"]


if __name__ == "__main__":
    unittest.main()
