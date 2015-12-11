"""
Unittests for likelihood calculation of discrete traits
"""
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discrete
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
        calculatedLikelihood = discrete.mk(tree, chars, Q)

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
        calculatedLikelihood = discrete.mk(tree, charstates, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkFlatroot_randtree10_matchesPhytools(self):
        charstates = self.randchars10
        tree = self.randTree10
        Q = self.randQ

        phytoolslogLikelihood = -8.298437
        calculatedLogLikelihood = discrete.mk(tree, charstates, Q)

        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedLogLikelihood))

    def test_mkFlatroot_randtree5_matchesPhytools(self):
        charstates = self.randchars5
        tree = self.randTree5
        Q = self.randQ

        phytoolslogLikelihood = -6.223166
        calculatedLogLikelihood = discrete.mk(tree, charstates, Q)



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

        calculatedLogLikelihood = discrete.mk(tree, charstates, Q,
                                                 pi ="Fitzjohn")
        self.assertTrue(np.isclose(expectedLikelihood, calculatedLogLikelihood))


    def test_mkMultiRegime_twoQ_matchesByHand(self):
        tree = self.threetiptree
        chars = self.charstates_011

        Q1 = np.array([[-0.1,0.1],[0.1,-0.1]], dtype=np.double)
        Q2 = np.array([[-0.2,0.2],[0.2,-0.2]], dtype=np.double)

        Qs = np.array([Q1,Q2])

        inds = np.array([[1,2,3],[4,]])

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

        P00D = 0.72466448
        P01D = 0.27533552
        P11D = 0.72466448
        P10D = 0.27533552

        L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
        L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)

        L0r = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
        L1r = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)

        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
        calculatedLikelihood = discrete.mk_multi_regime(tree, chars, Qs, locs = inds,
                                                        pi="Equal")

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkMultiRegime_twoQFourTip_matchesByHand(self):
        tree = ivy.tree.read("(((A:1,B:1)C:1,D:2)E:1,F:3)root;")
        chars = [0,1,1,0]

        Q1 = np.array([[-0.1,0.1],[0.1,-0.1]], dtype=np.double)
        Q2 = np.array([[-0.2,0.2],[0.2,-0.2]], dtype=np.double)
        Qs = np.array([Q1,Q2])
        inds = np.array([[1,2,3],[4,5,6]])
        L0A = 1;L1A = 0;L0B = 0;L1B = 1;L0D = 0;L1D = 1;L0F = 1;L1F = 0

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

        P00D = 0.72466448
        P01D = 0.27533552
        P11D = 0.72466448
        P10D = 0.27533552

        P00E = 0.83516002
        P01E = 0.16483998
        P11E = 0.83516002
        P10E = 0.16483998

        P00F = 0.65059711
        P01F = 0.34940289
        P11F = 0.65059711
        P10F = 0.34940289

        L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
        L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)

        L0E = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
        L1E = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)

        L0r = (P00E * L0E + P01E * L1E) * (P00F * L0F + P01F * L1F)
        L1r = (P10E * L0E + P11E * L1E) * (P10F * L0F + P11F * L1F)

        predictedLikelihood = math.log(L0r * 0.5 + L1r * 0.5)
        calculatedLikelihood = discrete.mk_multi_regime(tree, chars, Qs, locs = inds,
                                                        pi="Equal")

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))


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

        calculated = discrete.fitMk(tree, chars, Q="Equal", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="Equal", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="Sym", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="Sym", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equal")

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

        calculated = discrete.fitMk(tree, chars, Q="Equal", pi="Equilibrium")

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

        calculated = discrete.fitMk(tree, chars, Q="Equal", pi="Equilibrium")

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

        calculated = discrete.fitMk(tree, chars, Q="Sym", pi="Equilibrium")

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

        calculated = discrete.fitMk(tree, chars, Q="Sym", pi="Equilibrium")

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
    #     calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equilibrium")
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
    #     calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equilibrium")
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
    #     calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equilibrium")
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

        calculated = discrete.fitMk(tree, chars, Q)

        calcq = calculated["Q"]
        calclik = calculated["Log-likelihood"]

        self.assertTrue(np.array_equal(Q,calcq) & np.isclose(calclik, -34.7215))

    def test_fitMk_ARDQequilibriumpi_matchesPhytools(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        calculated = discrete.fitMk(tree, chars, Q="ARD", pi="Equilibrium")

    def test_fitMk_equalQ3traits_matchesPhytools(self):
        pass


    def test_qsd_ARDQ2_matchesphytools(self):
        Q = np.array([[-.3,.1,.2],[.05,-.3,.25],[.05,.1,-.15]], dtype = np.double)

        calculatedPi = discrete.qsd(Q)

        expectedPi = np.array([0.1428571,0.2500000,0.6071429])

        try: # Results will vary slightly from phytools (order of 1e-3)
             # Possibly due to differences in optimization implementation?
            np.testing.assert_allclose(expectedPi, calculatedPi, atol=1e-3)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_ERQ3_returnsflatpi(self):
        Q = np.array([[-.2,.2],[.1, -.1]], dtype = np.double)

        calculatedPi = discrete.qsd(Q)

        expectedPi = np.array([1.0/3.0, 2.0/3.0])

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_symmetricQ3_returnsflatpi(self):
        Q = np.array([[-.3,.1,.2],[.1,-.2,.1],[.2,.1,-.3]], dtype = np.double)

        calculatedPi = discrete.qsd(Q)

        expectedPi = np.array([1.0/3.0]*3)

        try:
            np.testing.assert_allclose(expectedPi, calculatedPi)
        except AssertionError:
            self.fail("expectedPi != calculatedPi")

    def test_qsd_ARDQ3_matchesphytools(self):
        Q = np.array([[-.3,.1,.2],[.05,-.3,.25],[.05,.1,-.15]], dtype = np.double)

        calculatedPi = discrete.qsd(Q)

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

        calculatedPi = discrete.qsd(Q)

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
            discrete.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("length of given pi does not match Q dimensions",
                              e.message)
    def test_mk_piNotSumTo1_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2
        Q = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])

        pi = np.array([0.0,0.0,0.0])

        try:
            discrete.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("values of given pi must sum to 1",
                              e.message)

    def test_mk_piWrongType_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2
        Q = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])

        pi = [0.0,0.0,1.0]

        try:
            discrete.mk(tree, chars, Q, pi=pi)
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("pi must be str or numpy array",
                              e.message)

    def test_fitMk_invalidQstring_raisesValueError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = "This is not a valid Q"

        try:
            discrete.fitMk(tree, chars, Q)
            self.fail("Value error not raised")
        except ValueError, e:
            self.assertEquals("Q str must be one of: 'Equal', 'Sym', 'ARD'",
                              e.message)


    def test_fitMk_invalidQMatrix_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.ones([3,3])

        try:
            discrete.fitMk(tree, chars, Q)
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("rows of q must sum to zero", e.message)

    def test_fitMk_invalidpi_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        try:
            discrete.fitMk(tree, chars, pi = "invalid pi")
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("Pi must be one of: 'Equal', 'Fitzjohn', 'Equilibrium'", e.message)


    def test_mkMultiRegime_sameQ_matchesmk(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.array([[-0.556216,0.278108,0.278108],
                      [0.278108,-0.556216,0.278108],
                      [0.278108,0.278108,-0.556216]])

        Qs = np.array([Q,Q])

        locs = [[i for i,n in enumerate(tree.postiter()) if not n.isroot][:50],
                 [i for i,n in enumerate(tree.postiter()) if not n.isroot][50:]]

        single = discrete.mk(tree, chars, Q)
        multi = discrete.mk_multi_regime(tree, chars, Qs, locs, pi="Equal")

        self.assertEquals(single, multi)

    # def test_createLikelihoodMulti_twoRtwoSER_correctresult(self):
    #     tree = self.randTree100Scale2
    #     chars = self.simChars100states3Scale2
    #     r1 = [ i for i,n in enumerate(tree.postiter()) if (not n in list(tree[15].postiter())) and (not n.isroot)]
    #     r2 = [ i for i in range(198) if not i in r1]
    #
    #     locs = [r1, r2]
    #
    #     lik = discrete.create_likelihood_function_multimk(tree, chars, Qtype="ER",
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

class Mk_hrm(mkMethods):
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

        calculatedLikelihood = discrete.hrm_mk(tree, chars, Q,2, pi="Equal")

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
        calculatedLikelihood = discrete.hrm_mk(tree, chars, Q, 3, pi="Equal")

        self.assertTrue(np.isclose(corHMMLik, calculatedLikelihood))
    def test_hrmMk_600tiptree_matchescorHMM(self):
        tree = ivy.tree.read("support/hrm_600tips.newick")
        Q = np.array([[-.06, .05, .01, 0],
               [.03, -.035, 0, .005],
               [.015, 0, -.315, .3],
                [0, .01, .2, -.21]])
        chars = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                    1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                    1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0,
                    0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1,
                    0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0,
                    1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
                    0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1,
                    1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                    0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1,
                    1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1,
                    1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0,
                    0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                    1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
                    1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
                    1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                    0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
                    0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0,
                    1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                    1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0,
                    1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
                    0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0]
        print discrete.hrm_mk(tree, chars, Q, 2, pi="Equal")

    def test_createHrmMkLikelihood_simpleLikelihood_createsproperfunction(self):
        tree = self.threetiptree
        chars = [0,1,1]
        Q = np.array([[-0.2,  0.1,  0.1,  0. ],
                   [ 0.1, -0.2,  0. ,  0.1],
                   [ 0.1,  0. , -0.2,  0.1],
                   [ 0. ,  0.1,  0.1, -0.2]])
        Qparams = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

        f = discrete.create_likelihood_function_hrm_mk(tree, chars, 2, "ARD")
        val = discrete.hrm_mk(tree, chars, Q, 2)

        self.assertTrue(np.isclose(f(Qparams), -1*val))





if __name__ == "__main__":
    unittest.main()
