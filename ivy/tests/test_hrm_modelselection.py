from __future__ import absolute_import, division, print_function, unicode_literals
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
    # def test_mod_232_validpartiallyconnected_returnsTrue(self):
    #     self.assertTrue(hrm_bayesian.is_valid_model(self.mod_232_validpartiallyconnected,
    #                                   2,6,3, self.modorder_232))
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
        self.qmat_mod_222 = [(0,1),(2,3),(4,5,6,7)]
        self.qmat_params_222 = range(8)
        self.qmat_222 = np.array([[-4,0,4,0],
                                  [1,-6,0,5],
                                  [6,0,-8,2],
                                  [0,7,3,-10]])

        self.qmat_mod_232 = [(0,1),(2,3),(4,5),tuple(range(6, 18))]
        self.qmat_params_232 = range(18)
        self.qmat_232 = np.array([[-16,0,6,0,10,0],
                                  [1,-19,0,7,0,11],
                                  [8,0,-24,2,14,0],
                                  [0,9,3,-27,0,15],
                                  [12,0,16,0,-32,4],
                                  [0,13,0,17,5,-35]])

        self.qmat_mod_322 = [(0,1,2,3,4,5),(6,7,8,9,10,11),(12,13,14,15,16,17)]
        self.qmat_params_322 = range(18)
        self.qmat_322 = np.array([[-13,0,1,12,0,0],
                                  [2,-18,3,0,13,0],
                                  [4,5,-23,0,0,14],
                                  [15,0,0,-28,6,7],
                                  [0,16,0,8,-33,9],
                                  [0,0,17,10,11,-38]])

        self.qmat_mod_242 = [(0,1),(2,3),(4,5),(6,7),tuple(range(8,32))]
        self.qmat_params_242 = tuple(range(32))
        self.qmat_242 = np.array([[-36,0,8,0,12,0,16,0],
                                 [1,-40,0,9,0,13,0,17],
                                 [10,0,-56,2,20,0,24,0],
                                 [0,11,3,-60,0,21,0,25],
                                 [14,0,22,0,-68,4,28,0],
                                 [0,15,0,23,5,-72,0,29],
                                 [18,0,26,0,30,0,-80,6],
                                 [0,19,0,27,0,31,7,-84]])
    def test_qmatfill_222(self):
        mod = self.qmat_mod_222
        Qparams = self.qmat_params_222
        Q = np.zeros([4,4])
        hrm_bayesian.fill_model_Q(mod,Qparams,Q)
        self.assertTrue(np.array_equal(self.qmat_222,Q))
    def test_qmatfill_232(self):
        mod = self.qmat_mod_232
        Qparams = self.qmat_params_232
        Q = np.zeros([6,6])
        hrm_bayesian.fill_model_Q(mod,Qparams,Q)
        self.assertTrue(np.array_equal(self.qmat_232,Q))
    def test_qmatfill_322(self):
        mod = self.qmat_mod_322
        Qparams = self.qmat_params_322
        Q = np.zeros([6,6])
        hrm_bayesian.fill_model_Q(mod,Qparams,Q)
        self.assertTrue(np.array_equal(self.qmat_322,Q))
    def test_qmatfill_242(self):
        mod = self.qmat_mod_242
        Qparams = self.qmat_params_242
        Q = np.zeros([8,8])
        hrm_bayesian.fill_model_Q(mod,Qparams,Q)
        self.assertTrue(np.array_equal(self.qmat_242,Q))
if __name__ == "__main__":
    unittest.main()
