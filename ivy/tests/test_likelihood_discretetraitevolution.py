"""
Unittests for likelihood calculation of discrete traits
"""
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discretetraits
import numpy as np
import math


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
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
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
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
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
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
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
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
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

        predictedLikelihood = L0r * 0.5 + L1r * 0.5
        calculatedLikelihood = discretetraits.mk(tree, chars, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    def test_mkFlatroo_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
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

        predictedLikelihood = 0.11279709097648889
        calculatedLikelihood = discretetraits.mk(tree, charstates, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_mkFlatroot_randtree10_matchesPhytools(self):
        charstates = self.randchars10
        tree = self.randTree10
        Q = self.randQ

        phytoolslogLikelihood = -8.298437
        calculatedLikelihood = discretetraits.mk(tree, charstates, Q)
        calculatedlogLikelihood = math.log(calculatedLikelihood)

        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedlogLikelihood))

    def test_mkFlatroot_randtree5_matchesPhytools(self):
        charstates = self.randchars5
        tree = self.randTree5
        Q = self.randQ

        phytoolslogLikelihood = -6.223166
        calculatedLikelihood = discretetraits.mk(tree, charstates, Q)
        calculatedlogLikelihood = math.log(calculatedLikelihood)


        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedlogLikelihood))

class estimateQMethods(unittest.TestCase):
    def setUp(self):
        self.randTree100 = ivy.tree.read("support/randtree100tips.newick")
        self.randTree100Scale2 = ivy.tree.read("support/randtree100tipsscale2.newick")

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
    def test_EqualRatesQ2traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states2 # Generated with a 2x2 Q matrix where alpha=beta=0.5

        expectedParam = np.array([[-0.4549581,0.4549581],[0.4549581,-0.4549581]])
        expectedLogLikelihood = -27.26863

        calculatedParam, calculatedLogLikelihood = discretetraits.fitMkER(tree, chars)

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam)
        except:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood,
                                   calculatedLogLikelihood))

    def test_EqualRatesQ3traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states3

        expectedParam = np.array([[-0.556216,0.278108,0.278108],
                                  [0.278108,-0.556216,0.278108],
                                  [0.278108,0.278108,-0.556216]])
        expectedLogLikelihood = -41.508675

        calculatedParam, calculatedLogLikelihood = discretetraits.fitMkER(tree, chars)

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-5)
        except:
            self.fail("expectedParam != calculatedParam")
        self.assertTrue(np.isclose(expectedLogLikelihood,
                                   calculatedLogLikelihood))

    def test_SymRatesQ3traits_matchesPhytools(self):

        tree = self.randTree100
        chars = self.simChars100states3

        expectedParam = np.array([[-.631001,0.462874,0.168128],
                        [0.462874,-0.462874,0.000000],
                        [0.168128,0.000000,-0.168128]])
        expectedLogLikelihood = -39.141458

        calculatedParam, calculatedLogLikelihood = discretetraits.fitMkSym(tree, chars)

        try:
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-5)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_ARDQ3traits_matchesPhytools(self):

        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        expectedParam = np.array([[-0.31973305, 0.136550, 0.183184],
                         [0.997779, -0.997779, 0.0000],
                         [3.315930, 0.0000, -3.315930]])

        expectedLogLikelihood = -32.697278

        calculatedParam, calculatedLogLikelihood = discretetraits.fitMkARD(tree, chars)

        try: # Need high tolerance for this test
            np.testing.assert_allclose(expectedParam, calculatedParam, atol = 1e-3)
        except AssertionError:
            self.fail("expectedParam != calculatedParam")

        self.assertTrue(np.isclose(expectedLogLikelihood, calculatedLogLikelihood))

    def test_fitMk_invalidQstring_raisesValueError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = "This is not a valid Q"

        try:
            discretetraits.fitMk(tree, chars, Q)
            self.fail("Value error not raised")
        except ValueError, e:
            self.assertEquals("Q str must be one of: 'Equal', 'Sym', 'ARD'",
                              e.message)


    def test_fitMk_invalidQMatrix_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.ones([3,3])

        try:
            discretetraits.fitMk(tree, chars, Q)
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("rows of q must sum to zero", e.message)

    def test_fitMk_invalidpi_raisesAssertionError(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        try:
            discretetraits.fitMk(tree, chars, pi = "invalid pi")
            self.fail("Assertion error not raised")
        except AssertionError, e:
            self.assertEquals("Pi must be one of: 'Equal', 'Fitzjohn'", e.message)

    def test_fitMk_fixedQ_returnsQandLogik(self):
        tree = self.randTree100Scale2
        chars = self.simChars100states3Scale2

        Q = np.array([[-.2,.1,.1],[.1,-.2,.1],[.1,.1,-.2]], dtype = np.double)

        calcq, calclik = discretetraits.fitMk(tree, chars, Q)

        self.assertTrue(np.array_equal(Q,calcq) & np.isclose(calclik, -34.7215))

if __name__ == "__main__":
    unittest.main()
