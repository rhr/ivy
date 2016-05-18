"""
Unittests for multiregime mk model
"""
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



if __name__ == "__main__":
    unittest.main()
