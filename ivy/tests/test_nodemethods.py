"""
Unittests for node methods
"""
import unittest
import ivy
import numpy as np

class tree_methods(unittest.Testcase):
    def setUp(self):
        self.primates = ivy.tree.read("../../examples/primates.newick")
        self.plants = ivy.tree.read("../../examples/plants.newick")


class tree_properties_methods(tree_methods):
    def test_apeNodeIdx_primatetree_correctvals(self):
        self.primates.ape_node_idx()
        trueIds = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        ids = [ n.apeidx for n in primates ]

        self.assertEquals(trueIds, ids)
    def test_mrca_primatesHomoPongo_returnsA(self):
        self.assertEquals(self.primates["A"],
             self.primates.mrca("Homo","Pongo"))
    def test_mrca_internalNodes_raisesException(self):
        try:
            self.primates.mrca("A", "B")

class IsSameTree_Methods(tree_methods):
    """
    Tests for the isSameTree method of ivy.tree.Node
    """
    def test_sameTreeSameLabels_returnsTrue(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = ivy.tree.read("../../examples/primates.newick")

        self.assertTrue(a.isSameTree(b))

    def test_difTrees_returnsFalse(self):
        a = ivy.tree.read("../../examples/primates.newick")
        b = ivy.tree.read("../../examples/plants.newick")

        self.assertFalse(a.isSameTree(b))
if __name__ == "__main__":
    unittest.main()
