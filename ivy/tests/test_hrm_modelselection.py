import ivy
import numpy as np
import math
import itertools
import collections
import scipy
import unittest

from ivy.chars import hrm_bayesian, hrm, mk

class ModelValidity(unittest.TestCase):
    def setUp(self):
        """
        Models to test
        """
        # format: mod_(nobschar)(nregime)(nparam)_(modelvalidity)
        modorder_list = itertools.product(range(2+1), repeat = 2**2-2)
        self.modorder_222 = {m:i for i,m in enumerate(modorder_list)}
        self.mod_222_valid = (1,1,2,1,1,1,1,1)
        self.mod_222_disconnected = (1,1,2,1,0,0,0,0)
        self.mod_222_missingparams = (1,1,1,1,1,1,1,1)
        self.mod_222_unorderedregimes = (1,1,0,0,2,1,1,1)
        self.mod_222_identicalregimes = (1,1,1,1,2,1,1,1)

        modorder_list = itertools.product(range(2+1), repeat = 2**2-2)
        self.modorder_232 = {m:i for i,m in enumerate(modorder_list)}
        self.mod_232_valid = (1,1,2,1,2,2,1,1,1,1,1,1,1,1,1,1,1,1)
        self.mod_232_disconnectedall = (1,1,2,1,2,2)+(0,)*12
        self.mod_232_disconnectedone = (1,1,2,1,2,2,1)+(0,)*11
        self.mod_232_validpartiallyconnected = (1,1,2,1,2,2)+(1,0,0,0,0,2,0,0,0,0,0,2)
        self.mod_232_missingparams = (1,0)*9
        self.mod_232_unorderedregimes = (2,2,1,1,0,0)+(1,)*12
        self.mod_232_identicalregimes = (0,0,2,2,2,2)+(1,)*12

        modorder_list = itertools.product(range(2+1), repeat = 3**2-3)
        self.modorder_322 = {m:i for i,m in enumerate(modorder_list)}
        self.mod_322_valid = (1,)*6 + (2,)*6 + (1,)*6
        self.mod_322_disconnected = (1,)*6 + (2,)*6 + (0,)*6
        self.mod_322_missingparams = (0,)*6 + (1,)*6 + (1,)*6
        self.mod_322_unorderedregimes = (2,)*6 + (1,)*6 + (1,)*6
        self.mod_322_identicalregimes = (2,)*6 + (2,)*6 + (1,)*6

        modorder_list = itertools.product(range(2+1), repeat = 2**2-2)
        self.modorder_242 = {m:i for i,m in enumerate(modorder_list)}
        self.mod_242_valid = (0,0) + (0,1) + (0,2) + (2,2) + (1,)*24
        self.mod_242_disconnected = (0,0) + (0,1) + (0,2) + (2,2) + (1,0,0,0) + (0,0,0,0) + (1,0,0,0) + (0,0,0,0) + (0,0,0,0) + (0,0,0,0)
        self.mod_242_validpartiallyconnected = (0,0) + (0,1) + (0,2) + (2,2) + (1,0,0,0) + (1,0,0,0) + (1,0,0,0) + (0,0,0,0) + (1,0,0,0) + (1,0,0,0)

    def test_mod_222_valid_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_222_valid,
                                      2,4,2, self.modorder_222))
    def test_mod_222_disconnected_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_222_disconnected,
                                      2,4,2, self.modorder_222))
    def test_mod_222_missingparams_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_222_missingparams,
                                      2,4,2, self.modorder_222))
    def test_mod_222_unorderedregimes_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_222_unorderedregimes,
                                      2,4,2, self.modorder_222))
    def test_mod_222_identicalregimes_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_222_identicalregimes,
                                      2,4,2, self.modorder_222))

    def test_mod_232_valid_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_232_valid,
                                      2,6,3, self.modorder_232))
    def test_mod_232_disconnectedall_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_232_disconnectedall,
                                      2,6,3, self.modorder_232))
    def test_mod_232_disconnectedone_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_232_disconnectedone,
                                      2,6,3, self.modorder_232))
    def test_mod_232_validpartiallyconnected_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_232_validpartiallyconnected,
                                      2,6,3, self.modorder_232))
    def test_mod_232_missingparams_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_232_missingparams,
                                      2,6,3, self.modorder_232))
    def test_mod_232_unorderedregimes_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_232_unorderedregimes,
                                      2,6,3, self.modorder_232))
    def test_mod_232_identicalregimes_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_232_identicalregimes,
                                      2,6,3, self.modorder_232))

    def test_mod_322_valid_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_322_valid,
                                      2,6,2, self.modorder_322))
    def test_mod_322_disconnected_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_322_disconnected,
                                      2,6,2, self.modorder_322))
    def test_mod_322_missingparams_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_322_missingparams,
                                      2,6,2, self.modorder_322))
    def test_mod_322_unordered_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_322_unorderedregimes,
                                      2,6,2, self.modorder_322))
    def test_mod_322_identicalregimes_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_322_identicalregimes,
                                      2,6,2, self.modorder_322))

    def test_mod_242_valid_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_242_valid,
                                      2,8,4, self.modorder_242))
    def test_mod_242_disconnected_returnsFalse(self):
        self.assertFalse(hrm_bayesian.is_valid_model(self.mod_242_disconnected,
                                      2,8,4, self.modorder_242))
    def test_mod_242_partiallyconnected_returnsTrue(self):
        self.assertTrue(hrm_bayesian.is_valid_model(self.mod_242_validpartiallyconnected,
                                      2,8,4, self.modorder_242))
class QmatrixFilling(unittest.TestCase):
    def setUp(self):
        pass


if __name__ == "__main__":
    unittest.main()
