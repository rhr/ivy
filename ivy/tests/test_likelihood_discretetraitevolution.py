"""
Unittests for likelihood calculation of discrete traits
"""
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discretetraits
import numpy as np


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

class TreelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1, 0.1], [0.2, -0.2]])
        self.Q3x3_sym = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])
        self.geospizaQ = np.array([[-0.6673543, 0.342223, 0],[0.3336771,-0.6673543, 0.3336771], [0.3336771, 0.3336771, -0.6673543]])
        self.geospizaQ3x3 = np.array([[-0.5, 0.25, 0.25],[0.25, -0.5, 0.25],[0.25, 0.25, -0.5]])

        self.charstates_011 = [0,1,1]
        self.charstates_01 = [0,1]
        self.geospizaChars = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]


        self.threetiptree = ivy.tree.read("((A:1,B:1)C:1,D:2)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")
        self.geospizaTree = ivy.tree.read("(((((((((fuliginosa:0.055,fortis:0.055):0.055,magnirostris:0.11):0.07333,conirostris:0.18333):0.00917,scandens:0.1925):0.0355,difficilis:0.228):0.10346,(pallida:0.08667,((parvulus:0.02,psittacula:0.02):0.015,pauper:0.035):0.05167):0.24479):0.13404,Platyspiza:0.4655):0.06859,fusca:0.53409):0.04924,Pinaroloxias:0.58333);")
    def tearDown(self):
        del(self.threetiptree)
    def test_treelikelihood_3tiptreeSymmetricQ2x2_returnslikelihood(self):
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

        predictedLikelihood = L0r + L1r
        calculatedLikelihood = discretetraits.treeLikelihood(tree, chars, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    def test_treelikelihood_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
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
        calculatedLikelihood = discretetraits.treeLikelihood(tree, charstates, Q)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

if __name__ == "__main__":
    unittest.main()
