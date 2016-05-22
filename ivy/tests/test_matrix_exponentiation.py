"""
Unittests for matrix exponentiation
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
import numpy as np

# class Dexpm_sliceMethods(unittest.TestCase):
#     def test_dexpmslice_oneQoneT2by2_returnsP(self):
#         """
#         Basic test case for exponentiating a matrix and storing it in a pre-allocated
#         array. 2 x 2 Q matrix
#         """
#         Q = np.array([[-1,1,],
#                       [1,-1,]], dtype=np.double, order="C")
#         t = 1.0 # Branch length
#
#         expectedP = np.array([[0.56766765, 0.43233236], # The expected value of e^Q
#                                    [0.43233236, 0.56766764]],
#                                     dtype=np.double, order="C")
#         i = 0 # Index of the p array we are supposed to use
#         p = np.empty([1, Q.shape[0], Q.shape[1]], dtype=np.double, order="C") # Empty p array to store values in
#
#         cyexpokit.dexpm_slice(Q, t, p, i) # Important to note that this changes p in place
#
#         self.assertTrue(np.allclose(expectedP, p))
#
#     def test_dexpmslice_oneQoneT4by4_returnsP(self):
#         """
#         Basic test case for exponentiating a matrix and storing it in a pre-allocated
#         array. 4 x 4 Q matrix.
#         """
#         Q = np.array([[-1,1,0,0],
#                       [0,-1,1,0],
#                       [0,0,-1,1],
#                       [0,0,0,0]], dtype=np.double, order='C')
#         t = 1.0 # Branch length
#
#         expectedP = np.array([[0.36787944, 0.36787944, 0.18393972, 0.0803014], # The expected value of e^Q*t
#                               [0, 0.36787944, 0.36787944,0.26424112],
#                               [0, 0, 0.36787944, 0.63212056],
#                               [0,0,0,1]],
#                                dtype=np.double, order="C")
#         i = 0 # Index of the p array we are supposed to use
#         p = np.empty([1, Q.shape[0], Q.shape[1]], dtype=np.double, order="C") # Empty p array to store values in
#
#         cyexpokit.dexpm_slice(Q, t, p, i) # Important to note that this changes p in place
#
#         try:
#             np.testing.assert_allclose(expectedP, p)
#         except:
#             self.fail


class Dexpm_treeMethods(unittest.TestCase):
        def test_dexpmtree_oneQtwoT2by2_returnsP(self):
            """
            Basic test case for exponentiating a matrix and storing it in a pre-allocated
            array. 2 x 2 Q matrix. List t of length 2
            """
            Q = np.array([[-1,1,],
                          [1,-1,]], dtype=np.double, order="C")
            t = np.array([1.0,1.0]) # Array of branch lengths

            expectedP = np.array([[[0.56766765, 0.43233236], # The expected value of e^Q*1
                                   [0.43233236, 0.56766764]],
                                   [[0.56766765, 0.43233236], # The expected value of e^Q*1
                                   [0.43233236, 0.56766764]]],
                                        dtype=np.double, order="C")
            p = cyexpokit.dexpm_tree(Q, t) # Important to note that this changes p in place

            try:
                np.testing.assert_allclose(expectedP, p)
            except:
                self.fail("expectedP != p")

        def test_dexpmtree_oneQtwoT2by2difT_returnsP(self):
            """
            Basic test case for exponentiating a matrix and storing it in a pre-allocated
            array. 2 x 2 Q matrix. List t of length 2
            """
            Q = np.array([[-1,1,],
                          [1,-1,]], dtype=np.double, order="C")
            t = np.array([1.0,2.0]) # Array of branch lengths

            expectedP = np.array([[[0.56766765, 0.43233236], # The expected value of e^Q*1
                                   [0.43233236, 0.56766764]],
                                   [[0.50915782, 0.49084218], # The expected value of e^Q*1
                                   [0.49084218, 0.50915782]]],
                                        dtype=np.double, order="C")
            p = cyexpokit.dexpm_tree(Q, t) # Important to note that this changes p in place

            try:
                np.testing.assert_allclose(expectedP, p)
            except:
                self.fail("expectedP != p")
        def test_dexpmtree_nonsquareQ_assertionerror(self):
            """
            Test that dexpm_tree returns an assertion error if given a non-square
            Q matrix.
            """
            Q = np.array([[-1,1], [-1,1], [1,-1]], dtype=np.double, order="C")
            t = np.array([1.0])


            try:
                cyexpokit.dexpm_tree(Q, t)
                self.fail()
            except AssertionError as e:
                self.assertEqual('q must be square', str(e))
        def test_dexpmtree_twithzeroes_assertionerror(self):
            """
            Test that dexpm_tree returns an assertion error if given a t-array
            with zero or negative values
            """
            Q = np.array([[-1,1,],
                          [1,-1,]], dtype=np.double, order="C")
            t = np.array([0.0, 1.0])

            try:
                cyexpokit.dexpm_tree(Q, t)
                self.fail()
            except AssertionError as e:
                self.assertEqual("All branch lengths must be greater than zero", str(e))






if __name__ == "__main__":
    unittest.main()
