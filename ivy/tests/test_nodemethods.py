"""
Unittests for node methods
"""
import unittest
import ivy
import numpy as np

class IsSameTree_Methods(unittest.TestCase):
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
