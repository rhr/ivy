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
    def setup(self):
        self.randTree10 = ivy.tree.read("support/randtree10tips.newick")
        self.randQSlow = np.array([[-0.01, 0.01], [0.01, -0.01]])
        self.randQMedium = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.randQFast = np.array([[-0.5, 0.5], [0.5, -0.5]])
