"""
Unittests for marginal ancestor reconstruction
"""
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discrete
import numpy as np
import math
import scipy


class NodelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])

        self.fourchars = [1,1,0,0]

        self.fourtiptree = ivy.tree.read("(((A:1,B:1)E:1,C:2)F:1,D:3)R;")
    def tearDown(self):
        del(self.fourtiptree)

    def test_ancrecon_4tiptree_matchesByHand(self):
        charstates = self.fourchars
        tree = self.fourtiptree
        Q = Q2x2_sym

        DL0A = 0;DL1A = 1;DL0B = 0;DL1B = 1;DL0C = 1;DL1C = 0;DL0D = 1;DL1D = 0

        P00A = 0.90936538
        P01A = 0.09063462
        P11A = 0.90936538
        P10A = 0.09063462

        P00B = 0.90936538
        P01B = 0.09063462
        P11B = 0.90936538
        P10B = 0.09063462

        P00C = 0.83516002
        P01C = 0.16483998
        P11C = 0.83516002
        P10C = 0.16483998

        P00D = 0.77440582
        P01D = 0.22559418
        P11D = 0.77440582
        P10D = 0.22559418

        P00E = 0.90936538
        P01E = 0.09063462
        P11E = 0.90936538
        P10E = 0.09063462

        P00F = 0.90936538
        P01F = 0.09063462
        P11F = 0.90936538
        P10F = 0.09063462

        # As calculated by known-correct Mk model
        DL0E = 0.0082146349699188971
        DL1E = 0.82694538804790085

        DL0F = 0.068833879485347305
        DL1F = 0.12408164996495177


        # Now we work up, starting with node F
        UL0R_m_F = (DL0D * P00D) + (DL1D * P01D)
        UL1R_m_F = (DL0D * P10D) + (DL1D * P11D)

        UL0F = DL0F*(UL0R_m_F * P00F + UL1R_m_F * P10F)
        UL1F = DL1F*(UL0R_m_F * P01F + UL1R_m_F * P11F)

        # Now node E

        # State 0
        UL0F_m_E_PD = (DL0C * P00C) + (DL1C * P01C)
        UL0F_m_E_PU = (UL0R_m_F*P00F) + (UL1R_m_F * P10F)

        UL1F_m_E_PD = (DL0C * P10C) + (DL1C * P11C)
        UL1F_m_E_PU = (UL0R_m_F*P01F) + (UL1R_m_F * P11F)

        UL0F_m_E =  UL0F_m_E_PD * UL0F_m_E_PU
        UL1F_m_E =  UL1F_m_E_PD * UL1F_m_E_PU

        # State 0
        UL0E = DL0E*(UL0F_m_E * P00E + UL1F_m_E * P10E)

        # State 1

        UL1E = DL1E*(UL0F_m_E * P01E + UL1F_m_E * P11E)
