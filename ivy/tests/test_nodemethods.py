"""
Unittests for node methods
"""
import unittest
import ivy
import numpy as np

class tree_methods(unittest.TestCase):
    def setUp(self):
        self.primates = ivy.tree.read("../../examples/primates.newick")
        self.plants = ivy.tree.read("../../examples/plants.newick")

        self.primatesBPoly = ivy.tree.read("support/primatesBPoly.newick")

class tree_properties_methods(tree_methods):
    ## Ape IDX
    def test_apeNodeIdx_primatetree_correctvals(self):
        self.primates.ape_node_idx()
        trueIds = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        ids = [ n.apeidx for n in self.primates ]
        self.assertEquals(trueIds, ids)

    ## MRCA
    def test_mrca_primatesHomoPongo_returnsA(self):
        self.assertEquals(self.primates["A"],
             self.primates.mrca("Homo","Pongo"))
    def test_mrca_internalNodes_returnsB(self):
        self.assertEquals(self.primates["B"],
                          self.primates.mrca("A", "B"))

    ## ismono
    def test_ismono_HomoPongoLabel_returnsTrue(self):
        mono = self.primates.ismono("Homo", "Pongo")
        self.assertTrue(mono)
    def test_ismono_HomoMacacaLabel_returnsFalse(self):
        mono = self.primates.ismono("Homo", "Macaca")
        self.assertFalse(mono)
    def test_ismono_HomoPongoLabelList_returnsTrue(self):
        mono = self.primates.ismono(["Homo", "Pongo"])
        self.assertTrue(mono)
    def test_ismono_HomoPongoNode_returnsTrue(self):
        tree = self.primates
        mono = self.primates.ismono(tree["Homo"], tree["Pongo"])
        self.assertTrue(mono)
    def test_ismono_HomoMacacaNode_returnsFalse(self):
        tree = self.primates
        mono = self.primates.ismono(tree["Homo"], tree["Macaca"])
        self.assertFalse(mono)
    def test_ismono_HomoPongoNodeList_returnsTrue(self):
        tree = self.primates
        mono = self.primates.ismono([tree["Homo"], tree["Pongo"]])
        self.assertTrue(mono)
    def test_ismono_Homo_raisesAssertionError(self):
        try:
            self.primates.ismono("Homo")
            self.fail
        except AssertionError, e:
            self.assertEquals(e.message[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoList_raisesAssertionError(self):
        try:
            self.primates.ismono(["Homo"])
            self.fail
        except AssertionError, e:
            self.assertEquals(e.message[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoNode_raisesAssertionError(self):
        try:
            self.primates.ismono(self.primates["Homo"])
            self.fail
        except AssertionError, e:
            self.assertEquals(e.message[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoNodeList_raisesAssertionError(self):
        try:
            self.primates.ismono([self.primates["Homo"]])
            self.fail
        except AssertionError, e:
            self.assertEquals(e.message[:36], "Need more than one leaf for ismono()")

    def test_ismono_HomoA_raisesAssertionError(self):
        try:
            self.primates.ismono("Homo", "A")
            self.self
        except AssertionError, e:
            self.assertEquals(e.message,
             "All given nodes must be leaves")
    def test_ismono_HomoAList_raisesAssertionError(self):
        try:
            self.primates.ismono(["Homo", "A"])
            self.self
        except AssertionError, e:
            self.assertEquals(e.message,
             "All given nodes must be leaves")
    def test_ismono_HomoANodes_raisesAssertionError(self):
        tree = self.primates
        try:
            self.primates.ismono(tree["Homo"], tree["A"])
            self.self
        except AssertionError, e:
            self.assertEquals(e.message,
             "All given nodes must be leaves")
    ## labeled
    def test_labeled_primates_returnsLabeledNodes(self):
        labelMethod = self.primates.labeled()
        trueLabels = ["root", "C", "B", "A", "Homo",
                      "Pongo", "Macaca", "Ateles",
                      "Galago"]
        labeledNodes = [self.primates[n] for n in trueLabels]
        self.assertEquals(labelMethod, labeledNodes)

    ## leaves
    def test_leaves_nofilter_returnsLeaves(self):
        trueLeaflabels = ["Homo", "Pongo", "Macaca", "Ateles", "Galago"]
        trueLeaves = [self.primates[n] for n in trueLeaflabels]
        self.assertEquals(self.primates.leaves(), trueLeaves)
    def test_leaves_simplefilter_returnsFilteredLeaves(self):
        def f(node):
            return "o" in node.label
        trueleafLabels = ["Homo","Pongo","Galago"]
        trueLeaves = [self.primates[n] for n in trueleafLabels]

        self.assertEquals(self.primates.leaves(f), trueLeaves)



class alterTreeMethods(tree_methods):
    def test_collapse_A_returnsBpolytomy(self):
        tree = self.primates
        tree["A"].collapse()

        self.primatesBPoly.treename = "primates"

        self.assertTrue(tree.is_same_tree(self.primatesBPoly, verbose=True))
    def test_collapse_root_returnsAssertionError(self):
        tree = self.primates
        try:
            tree.collapse()
            self.fail()
        except AssertionError:
            pass
    def test_collapse_addlength_returnsCorrectLength(self):
        tree = self.primates

        expectedLenHomo = tree["Homo"].length + tree["A"].length
        expectedLenPongo = tree["Pongo"].length + tree["A"].length

        tree["A"].collapse(add=True)

        treePoly = self.primatesBPoly

        treePoly["Homo"].length = expectedLenHomo
        treePoly["Pongo"].length = expectedLenPongo

        self.assertTrue(tree.is_same_tree(treePoly))

    def test_copy_copytree_returnsSameTree(self):
        tree = self.primates
        tree2 = tree.copy()

        self.assertTrue(tree.is_same_tree(tree2, check_id=True))
    def test_addNewChild_createsPolytomy(self):
        tree = self.primates

        newNode = ivy.tree.Node()
        newNode.label="N"

        tree["A"].append(newNode)



class is_same_tree_Methods(tree_methods):
    """
    Tests for the is_same_tree method of ivy.tree.Node
    """
    def test_sameTreeDifIDSignoreID_returnsTrue(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = ivy.tree.read("../../examples/primates.newick")

        self.assertTrue(a.is_same_tree(b))

    def test_sameTreeDifIDScheckID_returnsFalse(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = ivy.tree.read("../../examples/primates.newick")

        self.assertFalse(a.is_same_tree(b, check_id=True))

    def test_sameTreeSameIDScheckID_returnsTrue(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = a.copy()

        self.assertTrue(a.is_same_tree(b, check_id=True))

    def test_difTrees_returnsFalse(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = ivy.tree.read("../../examples/plants.newick")

        self.assertFalse(a.is_same_tree(b))
    def test_sameTreeLadderized_returnsTrue(self):
        """
        Unsure what behavior should be. Will return true for now
        """
        a = self.primates
        b = a.copy()

        b.ladderize()

        self.assertTrue(a.is_same_tree(b))


if __name__ == "__main__":
    unittest.main()
