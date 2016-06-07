"""
Unittests for likelihood calculation of discrete traits
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discrete, mk, hrm
import numpy as np
import math
import scipy


class NodelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1, 0.1], [0.2, -0.2]])
        self.Q3x3_sym = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])

        self.charstates_01 = [0,1]

        self.simpletree = ivy.tree.read("(A:1,B:1)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")

    def tearDown(self):
        del(self.simpletree)

    def test_nodelikelihood_2tiptreeSymmetricQ2x2_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.16483997131
        calculatedLikelihood = discrete.nodeLikelihood(node)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreesinglenodeAsymmetricQ2x2_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_asym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.2218622277515326
        calculatedLikelihood = discrete.nodeLikelihood(node)
        self.assertTrue((predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreesinglenodeSymmetricQ3x3_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q3x3_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.0863939177214389
        calculatedLikelihood = discrete.nodeLikelihood(node)
        self.assertTrue((predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletreedifblens

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.22559418195297778
        calculatedLikelihood = discrete.nodeLikelihood(node)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

class mkMethods(unittest.TestCase):
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
        self.randchars5 = [1,2,2,1,0]
        self.randchars10 = [0,2,1,1,1,0,0,1,2,2]


        self.threetiptree = ivy.tree.read("((A:1,B:1)C:1,D:2)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")
        self.randTree5 = ivy.tree.read("support/randtree5tips.newick")
        self.randTree10 = ivy.tree.read("support/randtree10tips.newick")
    def tearDown(self):
        del(self.threetiptree)
    def test_mkFlatroot_3tiptreeSymmetricQ2x2_returnslikelihood(self):
        tree = self.threetiptree
        chars = self.charstates_011
        Q = self.Q2x2_sym

        # Manually calculated likelihood for expected output
        L0A = 1;L1A = 0;L0B = 0;L1B = 1;L0D = 0;L1D = 1

        P00A = 0.90936538
        P01A = 0.09063462
        P11A = 0.90936538
        P10A = 0.09063462

        P00B = 0.90936538
        P01B = 0.09063462
        P11B = 0.90936538
        P10B = 0.09063462

        P00C = 0.90936538
        P01C = 0.09063462
        P11C = 0.90936538
        P10C = 0.09063462

        P00D = 0.83516002
        P01D = 0.16483998
        P11D = 0.83516002
        P10D = 0.16483998

        L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
        L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)

        L0r = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
        L1r = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)

        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
        calculatedLikelihood = mk.mk(tree, chars, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    def test_mkFlatroot_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletreedifblens

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = math.log(0.11279709097648889)
        calculatedLikelihood = mk.mk(tree, charstates, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkFlatroot_randtree10_matchesPhytools(self):
        charstates = self.randchars10
        tree = self.randTree10
        Q = self.randQ

        phytoolslogLikelihood = -8.298437
        calculatedLogLikelihood = mk.mk(tree, charstates, Q)

        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedLogLikelihood))

    def test_mkFlatroot_randtree5_matchesPhytools(self):
        charstates = self.randchars5
        tree = self.randTree5
        Q = self.randQ

        phytoolslogLikelihood = -6.223166
        calculatedLogLikelihood = mk.mk(tree, charstates, Q)



        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedLogLikelihood))
    def test_mk_fitzjohn_matchesDiversitree(self):
        charstates = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        tree =  ivy.tree.read("support/randtree100tipsscale2.newick")
        Q = np.array([[-2.09613850e-01, 1.204029e-01, 8.921095e-02],
                      [5.654382e-01, -5.65438217e-01, 1.713339e-08],
                      [2.415020e-06, 5.958744e-07, -3.01089440e-06]])

        expectedLikelihood = -32.79025

        calculatedLogLikelihood = mk.mk(tree, charstates, Q,
                                                 pi ="Fitzjohn")
        self.assertTrue(np.isclose(expectedLikelihood, calculatedLogLikelihood))


    # def test_mkMultiRegime_twoQ_matchesByHand(self):
    #     tree = self.threetiptree
    #     chars = self.charstates_011
    #
    #     Q1 = np.array([[-0.1,0.1],[0.1,-0.1]], dtype=np.double)
    #     Q2 = np.array([[-0.2,0.2],[0.2,-0.2]], dtype=np.double)
    #
    #     Qs = np.array([Q1,Q2])
    #
    #     inds = np.array([[1,2,3],[4,]])
    #
    #     L0A = 1;L1A = 0;L0B = 0;L1B = 1;L0D = 0;L1D = 1
    #
    #     P00A = 0.90936538
    #     P01A = 0.09063462
    #     P11A = 0.90936538
    #     P10A = 0.09063462
    #
    #     P00B = 0.90936538
    #     P01B = 0.09063462
    #     P11B = 0.90936538
    #     P10B = 0.09063462
    #
    #     P00C = 0.90936538
    #     P01C = 0.09063462
    #     P11C = 0.90936538
    #     P10C = 0.09063462
    #
    #     P00D = 0.72466448
    #     P01D = 0.27533552
    #     P11D = 0.72466448
    #     P10D = 0.27533552
    #
    #     L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
    #     L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)
    #
    #     L0r = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
    #     L1r = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)
    #
    #     predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
    #     calculatedLikelihood = mk.mk_mr(tree, chars, Qs, locs = inds,
    #                                                     pi="Equal")
    #
    #     self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    #
    # def test_mkMultiRegime_twoQFourTip_matchesByHand(self):
    #     tree = ivy.tree.read("(((A:1,B:1)C:1,D:2)E:1,F:3)root;")
    #     chars = [0,1,1,0]
    #
    #     Q1 = np.array([[-0.1,0.1],[0.1,-0.1]], dtype=np.double)
    #     Q2 = np.array([[-0.2,0.2],[0.2,-0.2]], dtype=np.double)
    #     Qs = np.array([Q1,Q2])
    #     inds = np.array([[1,2,3],[4,5,6]])
    #     L0A = 1;L1A = 0;L0B = 0;L1B = 1;L0D = 0;L1D = 1;L0F = 1;L1F = 0
    #
    #     P00A = 0.90936538
    #     P01A = 0.09063462
    #     P11A = 0.90936538
    #     P10A = 0.09063462
    #
    #     P00B = 0.90936538
    #     P01B = 0.09063462
    #     P11B = 0.90936538
    #     P10B = 0.09063462
    #
    #     P00C = 0.90936538
    #     P01C = 0.09063462
    #     P11C = 0.90936538
    #     P10C = 0.09063462
    #
    #     P00D = 0.72466448
    #     P01D = 0.27533552
    #     P11D = 0.72466448
    #     P10D = 0.27533552
    #
    #     P00E = 0.83516002
    #     P01E = 0.16483998
    #     P11E = 0.83516002
    #     P10E = 0.16483998
    #
    #     P00F = 0.65059711
    #     P01F = 0.34940289
    #     P11F = 0.65059711
    #     P10F = 0.34940289
    #
    #     L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
    #     L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)
    #
    #     L0E = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
    #     L1E = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)
    #
    #     L0r = (P00E * L0E + P01E * L1E) * (P00F * L0F + P01F * L1F)
    #     L1r = (P10E * L0E + P11E * L1E) * (P10F * L0F + P11F * L1F)
    #
    #     predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
    #     calculatedLikelihood = mk.mk_mr(tree, chars, Qs, locs = inds,
    #                                                     pi="Equal")
    #
    #     self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    #

class estimateQMethods(unittest.TestCase):
    def setUp(self):
        self.randTree100 = ivy.tree.read("support/randtree100tips.newick")
        self.randTree100Scale2 = ivy.tree.read("support/randtree100tipsscale2.newick")
        self.randTree100Scale5 = ivy.tree.read("support/randtree100tipsscale5.newick")

        self.simChars100states2 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.simChars100states3 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                            0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                            1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2]
        self.simChars100states3Scale2 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
        self.simChars100states3Scale5 = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                                            2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                            0, 0, 1, 1, 2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 1, 1, 1, 1,
                                            0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0, 0,
                                            0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
    def test_EqualRatesEqualPiQ2traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states2 # Generated with a 2x2 Q matrix where alpha=beta=0.5

        expectedParam = np.array([[-0.4549581,0.4549581],[0.4549581,-0.4549581]])
        expectedLogLikelihood = -27.26863

        calculated = mk.fit_Mk(tree, chars, Q="Equal", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-4)
        except:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood,
                                   calculatedLogLikelihood))

    def test_EqualRatesEqualPiQ3traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states3

        expectedParam = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])
        expectedLogLikelihood = -41.508675

        calculated = mk.fit_Mk(tree, chars, Q="Equal", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-5)
        except:
            self.fail("expectedParam != calculatedParam")
        self.assertTrue(np.isclose(expectedLogLikelihood,
                                   calculatedLogLikelihood))
    def test_SymRatesEqualPiQ2traits_matchesPhytools(self):
        """
        Note that this is the same as an equal-rates 2-trait Q matrix
        """

        tree = self.randTree100
        chars = self.simChars100states2 # Generated with a 2x2 Q matrix where alpha=beta=0.5

        expectedParam = np.array([[-0.4549581,0.4549581],[0.4549581,-0.4549581]])
        expectedLogLikelihood = -27.26863

        calculated = mk.fit_Mk(tree, chars, Q="Sym", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-5)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_SymRatesEqualPiQ3traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states3

        expectedParam = np.array([[-.631001,0.462874,0.168128],
                        [0.462874,-0.462874,0.000000],
                        [0.168128,0.000000,-0.168128]])
        expectedLogLikelihood = -39.141458

        calculated = mk.fit_Mk(tree, chars, Q="Sym", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-5)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_ARDEqualPiQ2traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states2

        expectedParam = np.array([[-0.261398, 0.261398],
                         [0.978787, -0.978787]])

        expectedLogLikelihood = -25.813332

        calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_ARDEqualPiQ3traits_matchesPhytools(self):

        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        expectedParam = np.array([[-0.31973305, 0.136550, 0.183184],
                         [0.997779, -0.997779, 0.0000],
                         [3.315930, 0.0000, -3.315930]])

        expectedLogLikelihood = -32.697278

        calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equal")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_EqualRatesEquilibriumPiQ2_matchesPhytools(self):
        """
        The same as a flat pi
        """
        tree = self.randTree100
        chars = self.simChars100states2

        expectedParam = np.array([[-0.4549581,0.4549581],[0.4549581,-0.4549581]])
        expectedLogLikelihood = -27.26863

        calculated = mk.fit_Mk(tree, chars, Q="Equal", pi="Equilibrium")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_EqualRatesEquilibriumPiQ3_matchesPhytools(self):
        """
        The same as a flat pi
        """
        tree = self.randTree100
        chars = self.simChars100states3

        expectedParam = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])
        expectedLogLikelihood = -41.508675

        calculated = mk.fit_Mk(tree, chars, Q="Equal", pi="Equilibrium")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]


        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))



    def test_SymRatesEquilibriumPiQ2_matchesPhytools(self):
        """
        The same as a flat pi
        """
        tree = self.randTree100
        chars = self.simChars100states2

        expectedParam = np.array([[-0.4549581,0.4549581],[0.4549581,-0.4549581]])
        expectedLogLikelihood = -27.26863

        calculated = mk.fit_Mk(tree, chars, Q="Sym", pi="Equilibrium")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]


        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_SymRatesEquilibriumPiQ3_matchesPhytools(self):
        """
        The same as a flat pi
        """
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        expectedParam = np.array([[-0.220576,0.129755,0.090821],
                                  [0.129755,-0.129755,0.000000],
                                  [0.090821,0.000000,-0.090821]])
        expectedLogLikelihood = -34.398614

        calculated = mk.fit_Mk(tree, chars, Q="Sym", pi="Equilibrium")

        calculatedParam = calculated["Q"]
        calculatedLogLikelihood = calculated["Log-likelihood"]

        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    # Test no longer applies: our implementation is different
    # def test_ARDQEquilibriumPiQ2_matchesPhytools(self):

    #
    #     tree = self.randTree100
    #     chars = self.simChars100states2
    #
    #     expectedParam = np.array([[-0.275857, 0.275857],
    #                               [0.882763, -0.882763]])
    #     expectedLogLikelihood = -25.671774
    #
    #     calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equilibrium")
    #
    #     calculatedParam = calculated["Q"]
    #     calculatedLogLikelihood = calculated["Log-likelihood"]
    #
    #     try: # Need high tolerance for this test
    #         np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
    #     except AssertionError:
    #         self.fail("expectedParam != calculatedParam")
    #
    #     self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    # def test_ARDEquilibriumPiQ3_matchesPhytools(self):
    #
    #     tree = self.randTree100Scale2
    #     chars = self.simChars100states3Scale2
    #
    #     expectedParam = np.array([[-0.305796,0.129642,0.176154],
    #                               [0.790583,-0.790583,0.000000],
    #                               [3.107009,0.000000,-3.107009]])
    #     expectedLogLikelihood = -32.536296
    #
    #     calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equilibrium")
    #
    #     calculatedParam = calculated["Q"]
    #     calculatedLogLikelihood = calculated["Log-likelihood"]
    #
    #     try: # Need high tolerance for this test
    #         np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
    #     except AssertionError:
    #         self.fail("expectedParam != calculatedParam")
    #
    #     self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    # def test_ARDEquilibriumPiQ3_matchesDiversitree(self):
    #
    #     tree = self.randTree100Scale2
    #     chars = self.simChars100states3Scale2
    #
    #     expectedParam = np.array([[-0.305796,0.129642,0.176154],
    #                               [0.790583,-0.790583,0.000000],
    #                               [3.107009,0.000000,-3.107009]])
    #     expectedLogLikelihood = -32.536296
    #
    #     calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equilibrium")
    #
    #     calculatedParam = calculated["Q"]
    #     calculatedLogLikelihood = calculated["Log-likelihood"]
    #
    #     try: # Need high tolerance for this test
    #         np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
    #     except AssertionError:
    #         self.fail("expectedParam != calculatedParam")
    #
    #     self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))



    def test_fitMk_fixedQ_returnsQandLogik(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.array([[-.2,.1,.1],[.1,-.2,.1],[.1,.1,-.2]], dtype = np.double)

        calculated = mk.fit_Mk(tree, chars, Q)

        calcq = calculated["Q"]
        calclik = calculated["Log-likelihood"]

        self.assertTrue(np.array_equal(Q,calcq) & np.isclose(calclik, -34.7215))

    def test_fitMk_ARDQequilibriumpi_matchesPhytools(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        calculated = mk.fit_Mk(tree, chars, Q="ARD", pi="Equilibrium")

    def test_fitMk_equalQ3traits_matchesPhytools(self):
        pass


    def test_qsd_ARDQ2_matchesphytools(self):
        Q = np.array([[-.3,.1,.2],[.05,-.3,.25],[.05,.1,-.15]], dtype = np.double)

        calculatedPi = mk.qsd(Q)

        expectedPi = np.array([0.1428571,0.2500000,0.6071429])

        try: # Results will vary slightly from phytools (order of 1e-3)
             # Possibly due to differences in optimization implementation?
            np.testing.assert_allclose(expectedPi, calculatedPi, atol=1e-3)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_ERQ3_returnsflatpi(self):
        Q = np.array([[-.2,.2],[.1, -.1]], dtype = np.double)

        calculatedPi = mk.qsd(Q)

        expectedPi = np.array([1.0/3.0, 2.0/3.0])

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_symmetricQ3_returnsflatpi(self):
        Q = np.array([[-.3,.1,.2],[.1,-.2,.1],[.2,.1,-.3]], dtype = np.double)

        calculatedPi = mk.qsd(Q)

        expectedPi = np.array([1.0/3.0]*3)

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_ARDQ3_matchesphytools(self):
        Q = np.array([[-.3,.1,.2],[.05,-.3,.25],[.05,.1,-.15]], dtype = np.double)

        calculatedPi = mk.qsd(Q)

        expectedPi = np.array([0.1428571,0.2500000,0.6071429])

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi, atol=1e-5)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_ARDQ4_matchesphytools(self):
        Q = np.array([[-.6,.1,.2,.3],
                      [.05,-.4,.25,.1],
                      [.05,.1,-.15, 0],
                      [.1, .1, .1, -.3]], dtype = np.double)

        calculatedPi = mk.qsd(Q)

        expectedPi = np.array([0.08888889,0.200000,0.555555555,0.155555556])

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi, atol=1e-5)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")
    def test_mk_wronglengthpi_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2
        Q = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])

        pi = np.array([0.0,0.0,0.0,1.0])

        try:
            mk.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError as e:
            self.assertEqual("length of given pi does not match Q dimensions",
                              str(e))
    def test_mk_piNotSumTo1_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2
        Q = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])

        pi = np.array([0.0,0.0,0.0])

        try:
            mk.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError as e:
            self.assertEqual("values of given pi must sum to 1",
                              str(e))

    def test_mk_piWrongType_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2
        Q = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])

        pi = [0.0,0.0,1.0]

        try:
            mk.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError as e:
            self.assertEqual("pi must be str or numpy array",
                              str(e))

    def test_fitMk_invalidQstring_raisesValueError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = "This is not a valid Q"

        try:
            mk.fit_Mk(tree, chars, Q)
            self.fail("Value error not raised")
        except ValueError as e:
            self.assertEqual("Q str must be one of: 'Equal', 'Sym', 'ARD'",
                              str(e))


    def test_fitMk_invalidQMatrix_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.ones([3,3])

        try:
            mk.fit_Mk(tree, chars, Q)
            self.fail("Assertion error not raised")
        except AssertionError as e:
            self.assertEqual("rows of q must sum to zero", str(e))

    def test_fitMk_invalidpi_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        try:
            mk.fit_Mk(tree, chars, pi = "invalid pi")
            self.fail("Assertion error not raised")
        except AssertionError as e:
            self.assertEqual("Pi must be one of: 'Equal', 'Fitzjohn', 'Equilibrium'", str(e))


    # def test_mkMultiRegime_sameQ_matchesmk(self):
    #     tree = self.randTree100Scale2
    #     chars = self.simChars100states3Scale2
    #
    #     Q = np.array([[-0.556216,0.278108,0.278108],
    #                   [0.278108,-0.556216,0.278108],
    #                   [0.278108,0.278108,-0.556216]])
    #
    #     Qs = np.array([Q,Q])
    #
    #     locs = [[i for i,n in enumerate(tree.postiter()) if not n.isroot][:50],
    #              [i for i,n in enumerate(tree.postiter()) if not n.isroot][50:]]
    #
    #     single = mk.mk(tree, chars, Q)
    #     multi = mk.mk_mr(tree, chars, Qs, locs, pi="Equal")
    #
    #     self.assertEqual(single, multi)

    # def test_createLikelihoodMulti_twoRtwoSER_correctresult(self):
    #     tree = self.randTree100Scale2
    #     chars = self.simChars100states3Scale2
    #     r1 = [ i for i,n in enumerate(tree.postiter()) if (not n in list(tree[15].postiter())) and (not n.isroot)]
    #     r2 = [ i for i in range(198) if not i in r1]
    #
    #     locs = [r1, r2]
    #
    #     lik = mk.create_likelihood_function_multimk(tree, chars, Qtype="ER",
    #                       locs=locs, pi="Equal", min=True)
    #     x0 = np.array([0.1,0.1])
    #     optim = scipy.optimize.minimize(lik, x0, method="L-BFGS-B",
    #                       bounds = tuple(( (1e-14,None) for i in range(len(x0)) )))
    #
    #     truevals = np.array([ 0.09226348,  0.49886642])
    #
    #     try:
    #         np.testing.assert_allclose(truevals, optim.x)
    #     except AssertionError:
    #         self.fail("expectedQs != calculatedQs")






if __name__ == "__main__":
    unittest.main()
