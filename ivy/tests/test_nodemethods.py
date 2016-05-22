"""
Unittests for node methods
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import unittest
import ivy
import numpy as np

class tree_methods(unittest.TestCase):
    def setUp(self):
        self.primates = ivy.tree.read("../../examples/primates.newick")
        self.plants = ivy.tree.read("../../examples/plants.newick")
        self.nicotiana = ivy.tree.read("../../examples/nicotiana.newick")

        self.primatesBPoly = ivy.tree.read("support/primatesBPoly.newick")
        self.primatesAPoly = ivy.tree.read("support/primatesAPoly.newick")

class basic_tree_methods(tree_methods):
    def test_asciitree(self):
        print(self.primates.ascii())
    def test_contains_true(self):
        t = self.primates
        self.assertTrue(t["A"] in t["C"])
    def test_contains_false(self):
        t = self.primates
        self.assertFalse(t["C"] in t["A"])
    def test_contains_selfTrue(self):
        t = self.primates
        self.assertTrue(t["A"] in t["A"])
    def test_len_returnslen(self):
        self.assertTrue(len(self.primates)==9)
    def test_preiter_returnsorder(self):
        t = self.primates
        preorder = list(self.primates.preiter())
        trueorder = [t["root"], t["C"],t["B"],
                    t["A"],t["Homo"],t["Pongo"],
                    t["Macaca"],t["Ateles"],t["Galago"]]
        self.assertEqual(preorder, trueorder)
    def test_postiter_returnsorder(self):
        t = self.primates
        postorder = list(self.primates.postiter())
        trueorder = [t["Homo"], t["Pongo"],t["A"],
                    t["Macaca"],t["B"],t["Ateles"],
                    t["C"],t["Galago"],t["root"]]
        self.assertEqual(postorder, trueorder)

    def test_children(self):
        children = self.primates["C"].children
        trueChildren = [self.primates["B"],self.primates["Ateles"]]
        self.assertEqual(children, trueChildren)
    def test_parent(self):
        parent = self.primates["B"].parent
        trueParent = self.primates["C"]
        self.assertEqual(parent,trueParent)
    def test_grep_ignorecase(self):
        t = self.primates
        found = t.grep("A")
        trueFound = [t["A"],t["Macaca"],t["Ateles"],t["Galago"]]
        self.assertEqual(found,trueFound)
    def test_grep_case(self):
        t = self.primates
        found = t.grep("A", ignorecase=False)
        trueFound = [t["A"],t["Ateles"]]
        self.assertEqual(found,trueFound)


class tree_properties_methods(tree_methods):
    ## Ape IDX
    def test_apeNodeIdx_primatetree_correctvals(self):
        self.primates.ape_node_idx()
        trueIds = [6, 7, 8, 9, 1, 2, 3, 4, 5]
        ids = [ n.apeidx for n in self.primates ]
        self.assertEqual(trueIds, ids)
    ## MRCA
    def test_mrca_primatesHomoPongo_returnsA(self):
        self.assertEqual(self.primates["A"],
             self.primates.mrca("Homo","Pongo"))
    def test_mrca_internalNodes_returnsB(self):
        self.assertEqual(self.primates["B"],
                          self.primates.mrca("A", "B"))

    ## ismono
    def test_ismono_HomoPongoLabel_returnsTrue(self):
        mono = self.primates.ismono("Homo", "Pongo")
        self.assertTrue(mono)
    def test_ismono_HomoMacacaLabel_returnsFalse(self):
        mono = self.primates.ismono("Homo", "Macaca")
        self.assertFalse(mono)
    def test_ismono_HomoPongoLabelList_returnsTrue(self):
        mono = self.primates.ismono(["Homo", "Pongo"])
        self.assertTrue(mono)
    def test_ismono_HomoPongoNode_returnsTrue(self):
        tree = self.primates
        mono = self.primates.ismono(tree["Homo"], tree["Pongo"])
        self.assertTrue(mono)
    def test_ismono_HomoMacacaNode_returnsFalse(self):
        tree = self.primates
        mono = self.primates.ismono(tree["Homo"], tree["Macaca"])
        self.assertFalse(mono)
    def test_ismono_HomoPongoNodeList_returnsTrue(self):
        tree = self.primates
        mono = self.primates.ismono([tree["Homo"], tree["Pongo"]])
        self.assertTrue(mono)
    def test_ismono_Homo_raisesAssertionError(self):
        try:
            self.primates.ismono("Homo")
            self.fail
        except AssertionError as e:
            self.assertEqual(str(e)[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoList_raisesAssertionError(self):
        try:
            self.primates.ismono(["Homo"])
            self.fail
        except AssertionError as e:
            self.assertEqual(str(e)[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoNode_raisesAssertionError(self):
        try:
            self.primates.ismono(self.primates["Homo"])
            self.fail
        except AssertionError as e:
            self.assertEqual(str(e)[:36], "Need more than one leaf for ismono()")
    def test_ismono_HomoNodeList_raisesAssertionError(self):
        try:
            self.primates.ismono([self.primates["Homo"]])
            self.fail
        except AssertionError as e:
            self.assertEqual(str(e)[:36], "Need more than one leaf for ismono()")

    def test_ismono_HomoA_raisesAssertionError(self):
        try:
            self.primates.ismono("Homo", "A")
            self.self
        except AssertionError as e:
            self.assertEqual(str(e),
             "All given nodes must be leaves")
    def test_ismono_HomoAList_raisesAssertionError(self):
        try:
            self.primates.ismono(["Homo", "A"])
            self.self
        except AssertionError as e:
            self.assertEqual(str(e),
             "All given nodes must be leaves")
    def test_ismono_HomoANodes_raisesAssertionError(self):
        tree = self.primates
        try:
            self.primates.ismono(tree["Homo"], tree["A"])
            self.self
        except AssertionError as e:
            self.assertEqual(str(e),
             "All given nodes must be leaves")
    ## labeled
    def test_labeled_primates_returnsLabeledNodes(self):
        labelMethod = self.primates.labeled()
        trueLabels = ["root", "C", "B", "A", "Homo",
                      "Pongo", "Macaca", "Ateles",
                      "Galago"]
        labeledNodes = [self.primates[n] for n in trueLabels]
        self.assertEqual(labelMethod, labeledNodes)

    ## leaves
    def test_leaves_nofilter_returnsLeaves(self):
        trueLeaflabels = ["Homo", "Pongo", "Macaca", "Ateles", "Galago"]
        trueLeaves = [self.primates[n] for n in trueLeaflabels]
        self.assertEqual(self.primates.leaves(), trueLeaves)
    def test_leaves_simplefilter_returnsFilteredLeaves(self):
        def f(node):
            return "o" in node.label
        trueleafLabels = ["Homo","Pongo","Galago"]
        trueLeaves = [self.primates[n] for n in trueleafLabels]

        self.assertEqual(self.primates.leaves(f), trueLeaves)
    def test_reindex_primates_assignscorrectpi(self):
        tree = self.primates
        tree2 = tree.copy()

        for t in tree:
            t.pi = 999
        tree.reindex()

        self.assertTrue([n.pi for n in tree] == [n.pi for n in tree2])



class alterTreeMethods(tree_methods):
    def test_collapse_A_returnsBpolytomy(self):
        tree = self.primates
        tree["A"].collapse()
        self.primatesBPoly.treename = "primates"
        self.assertTrue(tree.is_same_tree(self.primatesBPoly))
    def test_collapse_root_returnsAssertionError(self):
        tree = self.primates
        try:
            tree.collapse()
            self.fail("AssertionErrorNotRaised")
        except AssertionError as e:
            if str(e) == "AssertionErrorNotRaised":
                self.fail("AssertionErrorNotRaised")
    def test_collapse_addlength_returnsCorrectLength(self):
        tree = self.primates
        expectedLenHomo = tree["Homo"].length + tree["A"].length
        expectedLenPongo = tree["Pongo"].length + tree["A"].length
        tree["A"].collapse(add=True)
        treePoly = self.primatesBPoly
        treePoly["Homo"].length = expectedLenHomo
        treePoly["Pongo"].length = expectedLenPongo
        self.assertTrue(tree.is_same_tree(treePoly))


    def test_copy_copytree_returnsSameTree(self):
        tree = self.primates
        tree2 = tree.copy()
        self.assertTrue(tree.is_same_tree(tree2))


    def test_addChild_createsPolytomy(self):
        tree = self.primates
        tree2 = self.nicotiana
        newNode = tree2["Nicotiana_tomentosa"]
        tree["A"].add_child(newNode)

        tree.reindex()
        self.assertTrue(tree.is_same_tree(self.primatesAPoly))
    def test_addChild_childInChildren_assertionError(self):
        tree = self.primates
        try:
            tree["A"].add_child(tree["Homo"])
            self.fail("AssertionErrorNotRaised")
        except AssertionError as e:
            if str(e) == "AssertionErrorNotRaised":
                self.fail("AssertionErrorNotRaised")

    def test_bisectBranch_distance50_returnsKnee(self):
        tree = self.primates
        tree["A"].bisect_branch()
        kneeTree = ivy.tree.read("support/primatesAKnee.newick")
        tree.reindex()
        self.assertTrue(tree.is_same_tree(kneeTree))
    def test_bisectBranch_distance75_returnsKnee(self):
        tree = self.primates
        tree["A"].bisect_branch(distance=.75)
        kneeTree = ivy.tree.read("support/primatesAKnee2.newick")
        tree.reindex()
        self.assertTrue(tree.is_same_tree(kneeTree))
    def test_bisectBranch_root_assertionError(self):
        tree = self.primates
        try:
            tree.bisect_branch()
            self.fail("AssertionErrorNotRaised")
        except AssertionError as e:
            if str(e) == "AssertionErrorNotRaised":
                self.fail("AssertionErrorNotRaised")
    def test_bisectBranch_distancenegative_assertionError(self):
        tree = self.primates
        try:
            tree["A"].bisect_branch(-1.0)
            self.fail("AssertionErrorNotRaised")
        except AssertionError as e:
            if str(e) == "AssertionErrorNotRaised":
                self.fail("AssertionErrorNotRaised")

    def test_removeChild_removeChild_createsKnee(self):
        tree = self.primates
        tree["A"].remove_child(tree["Homo"])
        tree.reindex()
        treeAOneChild = ivy.tree.read("support/primatesAOneChild.newick")
        self.assertTrue(tree.is_same_tree(treeAOneChild))
    def test_removeChild_removeOnlyChild_createsLeaf(self):
        tree = self.primates
        tree["A"].remove_child(tree["Homo"])
        tree["A"].remove_child(tree["Pongo"])
        tree.reindex()
        self.assertTrue(tree["A"].isleaf)
        treeNoChild = ivy.tree.read("support/primatesANoChildren.newick")
        self.assertTrue(tree.is_same_tree(treeNoChild))
    def test_removeChild_notChild_raisesAssertionError(self):
        tree = self.primates
        try:
            tree["A"].remove_child(tree["B"])
            self.fail("AssertionError not raised")
        except AssertionError as e:
            self.assertEqual(str(e), "node 'B' not child of node 'A'")

    def test_dropTip_primatesDropOneTip_returnsTree(self):
        tree = self.primates
        tree2 = tree.drop_tip(["Homo"])
        trueTree = ivy.tree.read("support/primatesHomoDropped.newick")
        self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_primatesDropTwoTips_returnsTree(self):
        tree = self.primates
        tree2 = tree.drop_tip(["Homo", "Ateles"])
        trueTree = ivy.tree.read("support/primatesHomoAtelesDropped.newick")
        self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_plantsDropOneClade_returnsTree(self):
        tree = self.plants
        tree2 = tree.drop_tip(tree["Fabids"].leaves())

        trueTree = ivy.tree.read("support/plantsNoFabids.newick")

        self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_plantsDrop100randtips_returnsTree(self):
        tree = self.plants
        tipsToDrop = ["Vavilovia", "Dalbergiella", "Albizia", "Tripodion", "Balsaminaceae",
                    "Chapmannia", "Psophocarpus", "Thermopsis", "Maackia", "Brachyelytrum",
                    "Dahlstedtia", "Acacia", "Cornaceae", "Schefflerodendron", "Siparunaceae",
                    "Macadamia_jansenii", "Pinales", "Talinaceae", "Trochodendraceae",
                    "Uncaria", "Aspalathus", "Hamamelidaceae", "Collaea", "Suberanthus_neriifolius",
                    "Geoffroea", "Asteliaceae", "Angylocalyx", "Hicksbeachia", "Lansium",
                    "Ornithopus", "Eremosparton", "Grevillea", "Sartoria", "Cercis",
                    "Amorpha", "Erythrophleum", "Daviesia", "Coussarea", "Almaleea",
                    "Peteria", "Boryaceae", "Hindsia", "Lablab", "Nertera", "Brongniartia",
                    "Portlandia", "Lardizabalaceae", "Hydrangeaceae", "Dirachmaceae",
                    "Hamelia", "Desmodium", "Tessmannia", "Petalostylis", "Aphanamixis",
                    "Berberidopsidaceae", "Arachnothryx_leucophylla", "Merxmuelleraa",
                    "Bonnetiaceae", "Scorpiurus", "Avena", "Melianthaceae", "Alseuosmiaceae",
                    "Aeschynomene_b", "Camptosema", "Tropaeolaceae", "Erithalis",
                    "Stipagrostis", "Paeoniaceae", "Colchicaceae", "Oreophysa", "Lonchocarpus",
                    "Synoum", "Pseudosamanea", "Alexa", "Gynerium", "Liparia", "Sipaneopsis",
                    "Platycyamus", "Diplotropis", "Moringaceae", "Pseudoprosopis",
                    "Alzateaceae", "Harleyodendron", "Ophrestia", "Gardenia", "Pennisetum",
                    "Uraria", "Tamaricaceae", "Ammothamnus", "Tetragonolobus", "Baphia",
                    "Desmanthus", "Resedaceae", "Lythraceae", "Otoptera", "Phylacium",
                    "Ebenus", "Linderniaceae", "Jacksonia", "Platylobium"]
        tree2 = tree.drop_tip(tipsToDrop)
        trueTree = ivy.tree.read("support/plantsDrop100.newick")
        self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_plantsDropMultipleClades_returnsTree(self):
        tree = self.plants
        trueTree = ivy.tree.read("support/plants3CladesDropped.newick")
        toDrop = [n.leaves() for n in tree if n.label in ["Papilionoideae","Asterids", "Poaceae"]]
        toDrop = [n for s in toDrop for n in s]
        tree2 = tree.drop_tip(toDrop)
        self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_plantsDropNextToRoot_returnsTree(self):
        tree = self.plants
        trueTree = ivy.tree.read("support/plants_droppedmonilo.newick")
        tree2 = tree.drop_tip(["Monilophyte"])

        self.assertTrue(tree2.is_same_tree(trueTree))
    # def test_dropTip_plantsdropthenladderize_returnsTree(self):
    #     tree = self.plants
    #     trueTree = ivy.tree.read("support/plants3CladesDropped.newick")
    #     toDrop = [n.leaves() for n in tree if n.label in ["Papilionoideae","Asterids", "Poaceae"]]
    #     toDrop = [n for s in toDrop for n in s]
    #     tree2 = tree.drop_tip(toDrop)
    #     tree2.ladderize()
    #     self.assertTrue(tree2.is_same_tree(trueTree))
    def test_dropTip_polytomies_returnsTree(self):
        pass
    def test_dropTip_knees_returnsTree(self):
        pass
    def test_dropTip_kneesAndPolytomies_returnsTree(self):
        pass




# class is_same_tree_Methods(tree_methods):
#     """
#     Tests for the is_same_tree method of ivy.tree.Node
#     """
#     def test_sameTreeDifIDSignoreID_returnsTrue(self):
#         a = ivy.tree.read("../../examples/primates.newick")
#         b = ivy.tree.read("../../examples/primates.newick")
#
#         self.assertTrue(a.is_same_tree(b))
#
#     def test_sameTreeSameIDScheckID_returnsTrue(self):
#         a = ivy.tree.read("../../examples/primates.newick")
#         b = a.copy()
#
#         self.assertTrue(a.is_same_tree(b))
#
#     def test_difTrees_returnsFalse(self):
#         a = ivy.tree.read("../../examples/primates.newick")
#         b = ivy.tree.read("../../examples/plants.newick")
#
#         self.assertFalse(a.is_same_tree(b))
#     def test_sameTreeLadderized_returnsTrue(self):
#         """
#         Unsure what behavior should be. Will return true for now
#         """
#         a = self.primates
#         b = a.copy()
#
#         b.ladderize()
#
#         self.assertTrue(a.is_same_tree(b))
#

if __name__ == "__main__":
    unittest.main()
