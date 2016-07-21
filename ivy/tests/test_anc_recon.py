"""
Unittests for marginal ancestor reconstruction
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discrete
from ivy.chars import recon
import numpy as np
import math
import scipy


class NodelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1,0.1],[0.5,-0.5]])

        self.fourchars = [1,1,0,0]

        self.fourtiptree = ivy.tree.read(u"(((A:1,B:1)E:1,C:2)F:1,D:3)R;")
    def tearDown(self):
        del(self.fourtiptree)

    def test_ancrecon_4tiptree_matchesCorhmm(self):
        chars = self.fourchars
        tree = self.fourtiptree
        Q = self.Q2x2_sym

        out_cor = np.array([[0.68038206,0.3196179],
                            [0.59350514,0.4064949],
                            [0.05419418,0.9458058]])
        out = ivy.chars.recon.anc_recon_cat(tree, chars, Q)

        for i in range(2):
            for char in set(chars):
                self.assertTrue(np.isclose(
                                out_cor[i+1][char],
                                out[i+1][char]/np.sum(out[i+1][:2])))
    def test_ancrecon_4tiptreeAsym_matchesCormm(self):
        chars = self.fourchars
        tree = self.fourtiptree
        Q = self.Q2x2_asym

        out_cor =  np.array([[0.70557182, 0.2944282],
                             [0.54462274, 0.4553773],
                             [0.08647845, 0.9135216]])

        out = ivy.chars.recon.anc_recon_cat(tree, chars, Q)

        for i in range(2):
            for char in set(chars):
                self.assertTrue(np.isclose(
                                out_cor[i+1][char],
                                out[i+1][char]/np.sum(out[i+1][:2])))

    def test_ancrecon_mkmr(self):
        chars = self.fourchars
        tree = self.fourtiptree
        Q = np.array([self.Q2x2asym,self.Q2x2sym])
        switchpoints = [(tree["E"],0.33333333)]

        out = recon.anc_recon_mkmr(tree,chars,Q,switchpoints)

if __name__ == "__main__":
    unittest.main()
