"""
Unittests for multiregime mk model
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import mk, hrm, mk_mr
import numpy as np
import math
import scipy


class Mk_mr_tests(unittest.TestCase):
    def setUp(self):
        self.randTree10 = ivy.tree.read("support/randtree10tips.newick")
        self.randQSlow = np.array([[-0.01, 0.01], [0.01, -0.01]])
        self.randQMedium = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.randQFast = np.array([[-0.5, 0.5], [0.5, -0.5]])
    def test_locsfromswitch_oneswitch(self):
        tree = self.randTree10
        locs = mk_mr.locs_from_switchpoint(tree, [tree[5]])
        true_locs = [[5,6,7,8,9],[1,2,3,4,10,11,12,13,14,15,16,17,18]]
        self.assertTrue((locs==true_locs).all())

    def test_locsfromswitch_twoswitchsep(self):
        tree = self.randTree10
        locs = mk_mr.locs_from_switchpoint(tree, [tree[5], tree[14]])
        true_locs = [[5,6,7,8,9],[14,15,16,17,18],[1,2,3,4,10,11,12,13]]
        self.assertTrue((locs==true_locs).all())

    def test_locsfromswitch_twoswitchnested(self):
        tree = self.randTree10
        locs = mk_mr.locs_from_switchpoint(tree, [tree[5], tree[6]])
        true_locs = [[5,9],[6,7,8],[1,2,3,4,10,11,12,13,14,15,16,17,18]]
        self.assertTrue((locs==true_locs).all())

    def test_mkmr_matchesbyhand(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0]
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])
        locs = mk_mr.locs_from_switchpoint(tree,tree["C"])

        PA = [[ 0.449251  ,  0.550749  ],
              [ 0.367166  ,  0.632834  ]]

        PB = [[ 0.449251  ,  0.550749  ],
               [ 0.367166  ,  0.632834  ]]

        PC = [[ 0.449251  ,  0.550749  ],
               [ 0.367166  ,  0.632834  ]]

        PD = [[ 0.82721215,  0.17278785],
              [ 0.08639393,  0.91360607]]


        L0A = 0;L1A=1;L0B=1;L1B=0;L0D=1;L1D=0

        L0C = (PA[0][0] * L0A + PA[0][1] * L1A) * (PB[0][0] * L0B + PB[0][1] * L1B)
        L1C = (PA[1][0] * L0A + PA[1][1] * L1A) * (PB[1][0] * L0B + PB[1][1] * L1B)


        L0r = (PC[0][0] * L0C + PC[0][1] * L1C) * (PD[0][0] * L0D + PD[0][1] * L1D)
        L1r = (PC[1][0] * L0C + PC[1][1] * L1C) * (PD[1][0] * L0D + PD[1][1] * L1D)
        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
        calculatedLikelihood = mk_mr.mk_multi_regime(tree, chars, Qs, locs)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkmr_middleofbranch_matchesbyhand(self):
        tree = ivy.tree.read(u"((A:1,B:1)C:1,D:2)root;")
        chars = [1,0,0]
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])
        switchpoint = (tree["C"], 0.75)


        PA = [[ 0.449251  ,  0.550749  ],
              [ 0.367166  ,  0.632834  ]]#a

        PB = [[ 0.449251  ,  0.550749  ],
               [ 0.367166  ,  0.632834  ]]#b

        PCA = [[ 0.72115686,  0.27884314],
               [ 0.18589543,  0.81410457]]# Closer to tip

        PCB = [[ 0.9290649 ,  0.0709351 ],
               [ 0.03546755,  0.96453245]] # Closer to root

        PD = [[ 0.82721215,  0.17278785],
              [ 0.08639393,  0.91360607]]

        PCT = [[ 0.48751229,  0.51248771],
               [ 0.33980164,  0.66019836]]

        L0A = 0;L1A=1;L0B=1;L1B=0;L0D=1;L1D=0

        L0CA = (PA[0][0] * L0A + PA[0][1] * L1A) * (PB[0][0] * L0B + PB[0][1] * L1B)
        L1CA = (PA[1][0] * L0A + PA[1][1] * L1A) * (PB[1][0] * L0B + PB[1][1] * L1B)

        L0CB = PCA[0][0] * L0CA + PCA[0][1] * L1CA
        L1CB = PCA[1][0] * L0CA + PCA[1][1] * L1CA


        L0r_m = (PCB[0][0] * L0CB + PCB[0][1] * L1CB) * (PD[0][0] * L0D + PD[0][1] * L1D)
        L1r_m = (PCB[1][0] * L0CB + PCB[1][1] * L1CB) * (PD[1][0] * L0D + PD[1][1] * L1D)

        L0r_t = (PCT[0][0] * L0CA + PCT[0][1] * L1CA) * (PD[0][0] * L0D + PD[0][1] * L1D)
        L1r_t = (PCT[1][0] * L0CA + PCT[1][1] * L1CA) * (PD[1][0] * L0D + PD[1][1] * L1D)


        predictedLikelihood = math.log(L0r_m * 0.5 + L1r_m * 0.5)


if __name__ == "__main__":
    unittest.main()
