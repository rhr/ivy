from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discrete
import numpy as np
import math
import csv
import scipy



class ContrastMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.sol = ivy.tree.read("../../examples/solanaceae_sarkinen2013.newick")
    def test_PIC_bifurcatingtree_returncontrasts(self):
        tree = self.sol
        tree.length = 0.0 # Setting root length
        polvol = {}; stylen = {}
        with open("../../examples/pollenvolume_stylelength.csv", "r") as csvfile:
                    traits = csv.DictReader(csvfile, delimiter = str(","),
                                                     quotechar = str('"'))
                    for i in traits:
                        polvol[i["Binomial"]] = float(i["PollenVolume"])

        p = ivy.contrasts.PIC(tree, polvol)
        pic_pol =[p[key][2] for key in p.keys()]

        ape_picmean_notscaled = -0.06519213

        self.assertTrue(np.isclose(ape_picmean_notscaled, np.mean(pic_pol)))

class LTTMethods(unittest.TestCase):
    def setUp(self):
        self.primates = ivy.tree.read("../../examples/primates.newick")
    def test_ltt_primates_matchesApe(self):
        tree = self.primates
        l = ivy.ltt(tree)

        apeTimes = [0.00,0.38,0.51,0.79]
        apeNLin =  [1,2,3,4]

        self.assertTrue((apeTimes==l[0]).all())
        self.assertTrue((apeNLin==l[1]).all())

if __name__ == "__main__":
    unittest.main()
