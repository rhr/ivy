"""
Unittests for multiregime mk model
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import timeit
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars.cy_tree import cy_tree
from ivy.chars import mk, hrm, mk_mr
import numpy as np
import math
import scipy
np.set_printoptions(suppress=True, precision=5)

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

    # def test_mkmr_matchesbyhand(self):
    #     tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
    #     chars = [1,0,0,0]
    #     Q1 = np.array([[-0.10,0.10],
    #                    [0.05,-0.05]])
    #     Q2 = np.array([[-1.5,1.5],
    #                    [1.,-1.]])
    #     Qs = np.array([Q2,Q1])
    #     locs = mk_mr.locs_from_switchpoint(tree,[tree["C"]])
    #
    #     PA = [[ 0.449251  ,  0.550749  ],
    #           [ 0.367166  ,  0.632834  ]]
    #
    #     PB = [[ 0.449251  ,  0.550749  ],
    #            [ 0.367166  ,  0.632834  ]]
    #
    #     PC = [[ 0.449251  ,  0.550749  ],
    #            [ 0.367166  ,  0.632834  ]]
    #
    #     PD = [[ 0.82721215,  0.17278785],
    #           [ 0.08639393,  0.91360607]]
    #
    #     PE = [[ 0.90713865,  0.09286135],
    #           [ 0.04643067,  0.95356933]]
    #
    #     PF =[[ 0.75841877,  0.24158123],
    #          [ 0.12079062,  0.87920938]]
    #
    #
    #     L0A = 0;L1A=1;L0B=1;L1B=0;L0D=1;L1D=0;L0F=1;L1F=0
    #
    #     L0C = (PA[0][0] * L0A + PA[0][1] * L1A) * (PB[0][0] * L0B + PB[0][1] * L1B)
    #     L1C = (PA[1][0] * L0A + PA[1][1] * L1A) * (PB[1][0] * L0B + PB[1][1] * L1B)
    #
    #     L0E = (PC[0][0] * L0C + PC[0][1] * L1C) * (PD[0][0] * L0D + PD[0][1] * L1D)
    #     L1E = (PC[1][0] * L0C + PC[1][1] * L1C) * (PD[1][0] * L0D + PD[1][1] * L1D)
    #
    #     L0r = (PE[0][0] * L0E + PE[0][1] * L1E) * (PF[0][0] * L0F + PF[0][1] * L1F)
    #     L1r = (PE[1][0] * L0E + PE[1][1] * L1E) * (PF[1][0] * L0F + PF[1][1] * L1F)
    #
    #
    #     predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
    #     calculatedLikelihood = mk_mr.mk_mr(tree, chars, Qs, locs)
    #
    #     self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkmr_middleofbranch_matchesbyhand(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = {"A":1,"B":0,"D":0,"F":0}
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

        PCA = [[ 0.49201298,  0.50798702], # Closer to tip, fast regime
               [ 0.33865801,  0.66134199]]

        PCB = [[ 0.97546295,  0.02453705], # Closer to root, slow regime
               [ 0.01226853,  0.98773147]]

        PD = [[ 0.82721215,  0.17278785],
              [ 0.08639393,  0.91360607]]

        PE = [[ 0.90713865,  0.09286135],
              [ 0.04643067,  0.95356933]]

        PF =[[ 0.75841877,  0.24158123],
             [ 0.12079062,  0.87920938]]

        L0A = 0;L1A=1;L0B=1;L1B=0;L0D=1;L1D=0;L0F=1;L1F=0


        L0CA = (PA[0][0] * L0A + PA[0][1] * L1A) * (PB[0][0] * L0B + PB[0][1] * L1B)
        L1CA = (PA[1][0] * L0A + PA[1][1] * L1A) * (PB[1][0] * L0B + PB[1][1] * L1B)

        L0CB = PCA[0][0] * L0CA + PCA[0][1] * L1CA
        L1CB = PCA[1][0] * L0CA + PCA[1][1] * L1CA


        L0E = (PCB[0][0] * L0CB + PCB[0][1] * L1CB) * (PD[0][0] * L0D + PD[0][1] * L1D)
        L1E = (PCB[1][0] * L0CB + PCB[1][1] * L1CB) * (PD[1][0] * L0D + PD[1][1] * L1D)

        L0r = (PE[0][0] * L0E + PE[0][1] * L1E) * (PF[0][0] * L0F + PF[0][1] * L1F)
        L1r = (PE[1][0] * L0E + PE[1][1] * L1E) * (PF[1][0] * L0F + PF[1][1] * L1F)


        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
        calculatedLikelihood = mk_mr.mk_mr_midbranch(tree, chars, Qs, [switchpoint],debug=False)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))


    def test_mkmr_middleofbranchtwoswitch_matchesbyhand(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Q3 = np.array([[-.01,.01],
                       [.015,-.015]])
        Qs = np.array([Q2,Q3,Q1])
        switchpoint = [(tree["C"], 0.75),(tree["E"],0.25)]


        PA = [[ 0.449251  ,  0.550749  ],
              [ 0.367166  ,  0.632834  ]]#a

        PB = [[ 0.449251  ,  0.550749  ],
               [ 0.367166  ,  0.632834  ]]#b

        PCA = [[ 0.49201298,  0.50798702], # Closer to tip, fast regime
               [ 0.33865801,  0.66134199]]

        PCB = [[ 0.99749845,  0.0024922 ], # Closer to root, slow regime
               [ 0.00373829,  0.99625235]]

        PD =  [[ 0.98049177,  0.01950823],
               [ 0.02926235,  0.97073765]]

        PEA = [[ 0.99749845,  0.0024922 ],
               [ 0.00373829,  0.99625235]]

        PEB = [[ 0.9290649 ,  0.0709351 ],
               [ 0.03546755,  0.96453245]]

        PF = [[ 0.75841877,  0.24158123],
              [ 0.12079062,  0.87920938]]

        L0A = 0;L1A=1;L0B=1;L1B=0;L0D=1;L1D=0;L0F=1;L1F=0


        L0CA = (PA[0][0] * L0A + PA[0][1] * L1A) * (PB[0][0] * L0B + PB[0][1] * L1B)
        L1CA = (PA[1][0] * L0A + PA[1][1] * L1A) * (PB[1][0] * L0B + PB[1][1] * L1B)

        L0CB = PCA[0][0] * L0CA + PCA[0][1] * L1CA
        L1CB = PCA[1][0] * L0CA + PCA[1][1] * L1CA

        L0EA = (PCB[0][0] * L0CB + PCB[0][1] * L1CB) * (PD[0][0] * L0D + PD[0][1] * L1D)
        L1EA = (PCB[1][0] * L0CB + PCB[1][1] * L1CB) * (PD[1][0] * L0D + PD[1][1] * L1D)

        L0EB = PEA[0][0] * L0EA + PEA[0][1] * L1EA
        L1EB = PEA[1][0] * L0EA + PEA[1][1] * L1EA

        L0r = (PEB[0][0] * L0EB + PEB[0][1] * L1EB) * (PF[0][0] * L0F + PF[0][1] * L1F)
        L1r = (PEB[1][0] * L0EB + PEB[1][1] * L1EB) * (PF[1][0] * L0F + PF[1][1] * L1F)

        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)

        calculatedLikelihood = mk_mr.mk_mr_midbranch(tree, chars, Qs, switchpoint)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_likelihoodfunctionmods_correctlikelihood(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])

        lf = mk_mr.lf_mk_mr_midbranch_mods(tree, chars, mods=[(4,3),(2,1)], findmin=False)

        switchpoint = (tree["C"], 0.75)
        Qparams = np.array([0.05,0.1,1.0,1.5])
        lf_likelihood = lf(Qparams,[switchpoint])
        trueLikelihood =  mk_mr.mk_mr_midbranch(tree, chars, Qs, [switchpoint])
        self.assertTrue(np.isclose(lf_likelihood, trueLikelihood))

    def test_likelihoodfunction_correctlikelihood(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])

        lf = mk_mr.lf_mk_mr_midbranch(tree, chars,Qtype="ARD", nregime=2, findmin=False)

        switchpoint = (tree["C"], 0.75)
        Qparams = np.array([1.5,1.0,.1,.05])
        lf_likelihood = lf(Qparams,[switchpoint])
        trueLikelihood =  mk_mr.mk_mr_midbranch(tree, chars, Qs, [switchpoint])
        self.assertTrue(np.isclose(lf_likelihood, trueLikelihood))

    def test_maskarrayp_changeswitchpoint_correctentriesmasked(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]

        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])

        switchpoint_1 = [(tree["C"], 0.75)]
        ar = mk_mr.create_mkmr_mb_ar(tree, chars, 2)

        l1 = mk_mr.mk_mr_midbranch(tree, chars, Qs, switchpoint_1, ar=ar, debug=False)


        switchpoint_2 = [(tree["D"], 0.5)]

        l2 = mk_mr.mk_mr_midbranch(tree, chars, Qs, switchpoint_2, ar=ar, debug=False)


        ar2 = mk_mr.create_mkmr_mb_ar(tree, chars, 2)

        l2True = mk_mr.mk_mr_midbranch(tree, chars, Qs, switchpoint_2, ar=ar2, debug=False)

        self.assertEqual(l2, l2True)
    def test_maskarrayp_changeQ_correctentriesmasked(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]

        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Q3 = np.array([[-0.15,0.15],
                       [0.1,-0.1]])
        Qs1 = np.array([Q2,Q1])
        Qs2 = np.array([Q2,Q3])

        switchpoint = [(tree["C"], 0.75)]
        ar = mk_mr.create_mkmr_mb_ar(tree, chars, 2)

        l1 = mk_mr.mk_mr_midbranch(tree, chars, Qs1, switchpoint, ar=ar, debug=False)

        l2 = mk_mr.mk_mr_midbranch(tree, chars, Qs2, switchpoint, ar=ar, debug=False)

        ar2 = mk_mr.create_mkmr_mb_ar(tree, chars, 2)
        l2True = mk_mr.mk_mr_midbranch(tree, chars, Qs2, switchpoint, ar=ar2)

        self.assertEqual(l2, l2True)
    def test_maskarrayp_changeQswitch_correctentriesmasked(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]

        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Q3 = np.array([[-0.15,0.15],
                       [0.1,-0.1]])
        Qs1 = np.array([Q2,Q1])
        Qs2 = np.array([Q2,Q3])

        switchpoint_1 = [(tree["C"], 0.75)]
        switchpoint_2 = [(tree["D"], 0.5)]

        ar = mk_mr.create_mkmr_mb_ar(tree, chars, 2)

        l1 = mk_mr.mk_mr_midbranch(tree, chars, Qs1, switchpoint_1, ar=ar, debug=False)

        l2 = mk_mr.mk_mr_midbranch(tree, chars, Qs2, switchpoint_2, ar=ar, debug=False)

        ar2 = mk_mr.create_mkmr_mb_ar(tree, chars, 2)
        l2True = mk_mr.mk_mr_midbranch(tree, chars, Qs2, switchpoint_2, ar=ar2,debug=False)

        self.assertEqual(l2, l2True)
    def test_mkmr_benchmark(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        chars = [1,0,0,0]

        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Q3 = np.array([[-0.15,0.15],
                       [0.1,-0.1]])
        Qs1 = np.array([Q2,Q1])
        Qs2 = np.array([Q2,Q3])

        switchpoint_1 = [(tree["C"], 0.75)]
        switchpoint_2 = [(tree["D"], 0.5)]

        ar = mk_mr.create_mkmr_mb_ar(tree, chars, 2)
        def foo(tree=tree,chars=chars,Qs1=Qs1,switchpoint_1=switchpoint_1,ar=ar):
            mk_mr.mk_mr_midbranch(tree, chars, Qs1, switchpoint_1, ar=ar, debug=False)

        def dotime():
            t = timeit.Timer("foo()")
            time = t.timeit(1000)
            print("1000 loops took %fs\n" % (time,))

        import __builtin__
        __builtin__.__dict__.update(locals())

        dotime()
    def test_mkmr_largetree_correctlikelihood(self):
        tree = ivy.tree.read(u"support/hrm_600tips.newick")
        chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        Q0 = np.array([[-0.061457569587587413,0.061457569587587413],
                      [0.061457569587587413,-0.061457569587587413]])
        Q1 = np.array([[-1.4976115055655292,1.4976115055655292],
                      [1.4976115055655292,-1.4976115055655292]])
        Q2 = np.array([[-0.0014644343303779842,0.0014644343303779842],
                      [0.0014644343303779842,-0.0014644343303779842]])
        Qs = np.array([Q0,Q1,Q2])
        switchpoint0 = (tree[579],1.2)
        switchpoint1 = (tree[329],3.0)

        true_L = -89.213330113632566

        ar = mk_mr.create_mkmr_mb_ar(tree, chars, 3)

        calculated_l = mk_mr.mk_mr_midbranch(tree, chars, Qs, [switchpoint0,switchpoint1],ar=ar,debug=True)

        self.assertTrue(np.isclose(calculated_l, true_L))

    def test_makemklnlfunc_makesfunc(self):
        tree = ivy.tree.read(u"((t2:0.3778728602,(t3:0.03239763423,t4:0.03239763423):0.345475226):0.9831289164,t1:1.361001777);")
        chars = [0, 0, 0, 1]
        labels = [ lf.label for lf in tree.leaves() ]
        data = dict(zip(labels, chars))

        qidx = np.array(
            [[0,0,1,1],
             [0,1,0,1]],
            dtype=np.intp)

        f = cyexpokit.make_mklnl_func(tree, data, 2, 1, qidx)
        params = np.array([0.1,0.1])
        cylik = f(params,np.array([],dtype=int),np.array([]))

        Qs = np.array([[[-0.1,  0.1],
                     [ 0.1, -0.1]]])


        truelik = mk_mr.mk_mr_midbranch(tree, chars, Qs, [])

        self.assertTrue(np.isclose(cylik, truelik))
    def test_makemklnlfunc_multi_makesfunc(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        data = {"A":1,"B":0,"D":0,"F":0}
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Qs = np.array([Q2,Q1])
        switchpoint = (tree["C"], 0.75)

        qidx = np.array(
            [[0,0,1,1],
             [0,1,0,0],
             [1,0,1,3],
             [1,1,0,2]],
            dtype=np.intp)

        switches = np.array([switchpoint[0].ni], dtype=np.intp)
        lengths = np.array([switchpoint[1]], dtype=np.double)


        switches = np.array([tree[switchpoint[0].id].ni], dtype=np.intp)
        lengths = np.array([switchpoint[1]], dtype=np.double)

        f = cyexpokit.make_mklnl_func(tree, data, 2, 2, qidx)

        params = np.array([0.05,0.1,1.0,1.5])
        cylik = f(params, switches, lengths)

        truelik = mk_mr.mk_mr_midbranch(tree, data, Qs, [switchpoint])


        trueinds = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0])

        self.assertTrue((trueinds==f.qi).all())

        self.assertTrue(np.isclose(cylik, truelik))
    def test_makemklnlfunc_twoswitch_makesfunc(self):
        tree = ivy.tree.read(u'(((A:1,B:1)C:1,D:2)E:1,F:3)root;')
        data = {"A":1,"B":0,"D":0,"F":0}
        Q1 = np.array([[-0.10,0.10],
                       [0.05,-0.05]])
        Q2 = np.array([[-1.5,1.5],
                       [1.,-1.]])
        Q3 = np.array([[-.01,.01],
                       [.015,-.015]])
        Qs = np.array([Q2,Q3,Q1])
        switchpoint = [(tree["C"], 0.75),(tree["E"],0.25)]

        qidx = np.array(
            [[0,0,1,1],
             [0,1,0,0],
             [1,0,1,3],
             [1,1,0,2],
             [2,0,1,4],
             [2,1,0,5]],
            dtype=np.intp)

        switches = np.array([s[0].ni for s in switchpoint], dtype=np.intp)
        lengths = np.array([s[1] for s in switchpoint], dtype=np.double)

        f = cyexpokit.make_mklnl_func(tree, data, 2, 3, qidx)

        params = np.array([0.05,0.1,1.0,1.5,.01,.015])
        cylik = f(params, switches, lengths)

        truelik = mk_mr.mk_mr_midbranch(tree, data, Qs, switchpoint)
        self.assertTrue(np.isclose(cylik, truelik))

        # If we re-use the same likelihood function, do we get the correct
        # likelihood?
        switches_2 = np.array([2,6],dtype=np.int)
        lens_2 = np.array([.333,.5])

        f_2 = cyexpokit.make_mklnl_func(tree, data, 2, 3, qidx)

        cylik1 = f(params, switches_2,lens_2)
        cylik2 = f_2(params, switches_2, lens_2)
        self.assertTrue(np.isclose(cylik1,cylik2))

    def test_makemklnl_largetree(self):
        tree = ivy.tree.read(u"support/hrm_600tips.newick")
        chars = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0,
        1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
        1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        data = dict(zip([n.label for n in tree.leaves()],chars))
        Q0 = np.array([[-0.061457569587587413,0.061457569587587413],
                      [0.061457569587587413,-0.061457569587587413]])
        Q1 = np.array([[-1.4976115055655292,1.4976115055655292],
                      [1.4976115055655292,-1.4976115055655292]])
        Q2 = np.array([[-0.0014644343303779842,0.0014644343303779842],
                      [0.0014644343303779842,-0.0014644343303779842]])
        Qs = np.array([Q0,Q1,Q2])
        switchpoint0 = (tree[579],1.2)
        switchpoint1 = (tree[329],3.0)
        switchpoint = [switchpoint0,switchpoint1]

        true_L = -89.213330113632566
        qidx = np.array(
            [[0,0,1,0],
             [0,1,0,0],
             [1,0,1,1],
             [1,1,0,1],
             [2,0,1,2],
             [2,1,0,2]],
            dtype=np.intp)

        switches = np.array([tree[s[0].id].ni for s in switchpoint], dtype=np.intp)
        lengths = np.array([s[1] for s in switchpoint], dtype=np.double)

        f = cyexpokit.make_mklnl_func(tree, data, 2, 3, qidx)
        cy_l = f(np.array([0.0014644,0.061457,1.497611]), switches, lengths)

        self.assertTrue(np.isclose(true_L, cy_l, atol=1e-4))

        switches = np.array([200, 300])
        lengths = np.array([.3,1.2])
        cy_l = f(np.array([0.01,0.061457,1.497611]), switches, lengths)

        f2 = cyexpokit.make_mklnl_func(tree, data, 2, 3, qidx)
        true_L = f2(np.array([0.01,0.061457,1.497611]), switches, lengths)
        self.assertTrue(np.isclose(true_L, cy_l, atol=1e-4))


    def test_cytree_makestree(self):
        tree = ivy.tree.read(u"support/hrm_600tips.newick")
        cytree = cy_tree.tree_from_ivy(tree)




if __name__ == "__main__":
    unittest.main()
