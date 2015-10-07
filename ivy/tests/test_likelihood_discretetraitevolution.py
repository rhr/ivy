"""
Unittests for likelihood calculation of discrete traits
"""
import unittest
import ivy
from ivy.chars.expokit import cyexpokit
from ivy.chars import discretetraits
import numpy as np
import math


class NodelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1, 0.1], [0.2, -0.2]])
        self.Q3x3_sym = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])

        self.charstates_01 = [0,1]

        self.simpletree = ivy.tree.read("(A:1,B:1)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")

    def tearDown(self):
        del(self.simpletree)

    def test_nodelikelihood_2tiptreeSymmetricQ2x2_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.16483997131
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreesinglenodeAsymmetricQ2x2_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_asym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.2218622277515326
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
        self.assertTrue((predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreesinglenodeSymmetricQ3x3_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletree

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q3x3_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.0863939177214389
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
        self.assertTrue((predictedLikelihood, calculatedLikelihood))

    def test_nodelikelihood_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletreedifblens

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.22559418195297778
        calculatedLikelihood = discretetraits.nodeLikelihood(node)
        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

class TreelikelihoodMethods(unittest.TestCase):
    def setUp(self):
        """
        Define objects to be used in tests
        """
        self.Q2x2_sym = np.array([[-0.1, 0.1], [0.1, -0.1]])
        self.Q2x2_asym = np.array([[-0.1, 0.1], [0.2, -0.2]])
        self.Q3x3_sym = np.array([[-0.1, 0.05, 0.05], [0.05, -0.1, 0.05], [0.05, 0.05, -0.1]])
        self.randQ = np.array([[-2,1,1],[1,-2,1],[1,1,-2]], dtype=np.double)

        self.charstates_011 = [0,1,1]
        self.charstates_01 = [0,1]
        self.randchars = [0, 0, 0, 0, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0,0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 2, 0, 0,0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 2, 1,2, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0]


        self.threetiptree = ivy.tree.read("((A:1,B:1)C:1,D:2)root;")
        self.simpletreedifblens = ivy.tree.read("(A:1,B:2)root;")
        self.randTree = ivy.tree.read("((t19:0.1455370501,(t93:0.003485132701,t94:0.003485132701):0.1420519174):0.8544629499,(((t20:0.137695654,t21:0.137695654):0.01470363979,(t69:0.02319238016,t70:0.02319238016):0.1292069137):0.7706028377,((((((t9:0.2058689769,t10:0.2058689769):0.02445824439,(t14:0.1737959407,((t83:0.01065806023,t84:0.01065806023):0.121469687,((t85:0.01029525711,t86:0.01029525711):0.005478724299,(t87:0.008807231647,t88:0.008807231647):0.006966749761):0.1163537658):0.0416681935):0.05653128062):0.06982634285,(t27:0.1090130745,t28:0.1090130745):0.1911404897):0.1787158792,(((t5:0.258753861,t6:0.258753861):0.1410300566,((t81:0.01591730853,t82:0.01591730853):0.3615724363,((t30:0.0996702128,(t33:0.07905766289,((t99:0.0004570989347,t100:0.0004570989347):0.03224475899,t62:0.03270185792):0.04635580497):0.02061254992):0.1276043309,(t75:0.01894359348,t76:0.01894359348):0.2083309502):0.1502152011):0.02229417278):0.04748194751,((((t65:0.0281660328,t66:0.0281660328):0.01576391803,(t74:0.01971899718,(t95:0.002215152112,t96:0.002215152112):0.01750384507):0.02421095365):0.02175481795,t40:0.06568476878):0.3583509436,(((t77:0.01858388195,t78:0.01858388195):0.2113898101,t8:0.2299736921):0.06822726214,((t72:0.02099923629,t73:0.02099923629):0.1390365793,t18:0.1600358156):0.1381651386):0.1258347582):0.02323015273):0.03160357824):0.06907507411,((t67:0.02631208289,t68:0.02631208289):0.3343605894,((((t97:0.0005396284902,t98:0.0005396284902):0.05566029255,t45:0.05619992104):0.01970931922,(t47:0.05112861576,(t63:0.03039444455,t64:0.03039444455):0.02073417121):0.0247806245):0.02738104125,t29:0.1032902815):0.2573823908):0.1872718452):0.2391287856,(((t1:0.3656425279,((((t41:0.06056108219,t42:0.06056108219):0.01128628925,t37:0.07184737144):0.1614255279,t7:0.2332728994):0.01711002245,((t48:0.04864005553,t49:0.04864005553):0.1785961908,((t11:0.1755746135,((t50:0.04757269996,(t60:0.03552907531,t61:0.03552907531):0.01204362465):0.06428624887,t24:0.1118589488):0.06371566462):0.03874589225,((((t32:0.08753121452,(t43:0.05891303826,t44:0.05891303826):0.02861817626):0.001001815659,t31:0.08853303018):0.004897794044,(t51:0.04708779412,t52:0.04708779412):0.04634303011):0.07773445578,t15:0.17116528):0.0431552257):0.01291574059):0.02314667553):0.115259606):0.09641181683,((((t16:0.1673987831,((t91:0.005878472743,t92:0.005878472743):0.1065180698,t23:0.1123965425):0.05500224054):0.03181011113,(t89:0.008356630632,t90:0.008356630632):0.1908522636):0.0692985232,t4:0.2685074174):0.02251147663,(t12:0.1755065276,(t25:0.1096943278,t26:0.1096943278):0.06581219982):0.1155123664):0.1710354507):0.1774343655,(((((t35:0.07415763035,t36:0.07415763035):0.0204418025,(t38:0.06646251262,t39:0.06646251262):0.02813692023):0.07987630309,t13:0.1744757359):0.1221274344,(t56:0.03905360404,t57:0.03905360404):0.2575495663):0.293403109,(((t2:0.3262080045,t3:0.3262080045):0.005357133394,(t22:0.1141437974,(t55:0.04366106584,(t58:0.03642046608,t59:0.03642046608):0.007240599757):0.07048273159):0.2174213405):0.1617281294,((((t53:0.04654292462,t54:0.04654292462):0.005121109445,t46:0.05166403407):0.1139222414,t17:0.1655862755):0.116357162,(((t79:0.01657990811,t80:0.01657990811):0.005036964763,t71:0.02161687288):0.05670757672,t34:0.0783244496):0.2036189879):0.2113498298):0.09671301208):0.04948243081):0.1475845928):0.1359288285):0.07699786848);")
    def tearDown(self):
        del(self.threetiptree)
    def test_treelikelihood_3tiptreeSymmetricQ2x2_returnslikelihood(self):
        tree = self.threetiptree
        chars = self.charstates_011
        Q = self.Q2x2_sym

        # Manually calculated likelihood for expected output
        L0A = 1;L1A = 0;L0B = 0;L1B = 1;L0D = 0;L1D = 1

        P00A = 0.90936538
        P01A = 0.09063462
        P11A = 0.90936538
        P10A = 0.09063462

        P00B = 0.90936538
        P01B = 0.09063462
        P11B = 0.90936538
        P10B = 0.09063462

        P00C = 0.90936538
        P01C = 0.09063462
        P11C = 0.90936538
        P10C = 0.09063462

        P00D = 0.83516002
        P01D = 0.16483998
        P11D = 0.83516002
        P10D = 0.16483998

        L0C = (P00A * L0A + P01A * L1A) * (P00B * L0B + P01B * L1B)
        L1C = (P10A * L0A + P11A * L1A) * (P10B * L0B + P11B * L1B)

        L0r = (P00C * L0C + P01C * L1C) * (P00D * L0D + P01D * L1D)
        L1r = (P10C * L0C + P11C * L1C) * (P10D * L0D + P11D * L1D)

        predictedLikelihood = L0r * 0.5 + L1r * 0.5
        calculatedLikelihood = discretetraits.treeLikelihood(tree, chars, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))
    def test_treelikelihood_2tiptreeSymmetricQ2x2difblens_returnslikelihood(self):
        charstates = self.charstates_01
        tree = self.simpletreedifblens

        for i, node in enumerate(tree.leaves()):
            node.charstate = charstates[i]

        Q = self.Q2x2_sym
        t = [ node.length for node in tree.descendants() ]
        t = np.array(t, dtype=np.double)
        p = cyexpokit.dexpm_tree(Q,t)

        for i, node in enumerate(tree.descendants()):
            node.pmat = p[i]
        node = tree

        predictedLikelihood = 0.11279709097648889
        calculatedLikelihood = discretetraits.treeLikelihood(tree, charstates, Q)

        self.assertTrue(np.isclose(predictedLikelihood, calculatedLikelihood))

    def test_treelikelihood_randtree_matchesPhytools(self):
        charstates = self.randchars
        tree = self.randTree
        Q = self.randQ

        phytoolslogLikelihood = -62.096802
        calculatedLikelihood = discretetraits.treeLikelihood(tree, charstates, Q)
        calculatedlogLikelihood = math.log(calculatedLikelihood)

        self.assertTrue(np.isclose(phytoolslogLikelihood, calculatedlogLikelihood))

if __name__ == "__main__":
    unittest.main()
