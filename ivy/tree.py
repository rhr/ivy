"""
The Node class and functions for creating trees from Newick strings,
etc.

ivy does not have a Tree class per se, as most functions operate
directly on Node objects.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict
import os, types
import csv
import itertools


from .storage import Storage
from copy import copy as _copy
from .matrix import vcv
import ivy.newick
import numpy as np


try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]

def traverse(node):
    "recursive preorder iterator based solely on .children attribute"
    yield node
    for child in node.children:
        for descendant in traverse(child):
            yield descendant

class Node(object):
    """
    A basic Node class with attributes and references to child nodes
    ('children', a list) and 'parent'.

    Keyword Args:
        id: ID of the node. If not provided, is set using
          builtin id function
        ni (int): Node index.
        li (int): Leaf index.
        isroot (bool): Is the node a root.
        isleaf (bool): Is the node a leaf.
        label (str): Node label.
        length (float): Branch length from node to parent
        support: Bootstrap support values
        age (float): Age of the node in time units.
        parent (Node): Parent of the ndoe.
        children (list): List of node objects. Children of node
        nchildren (int): Number of children
        left: The "left" node
        treename: Name of tree
        comment: Comments for tree

    """
    def __init__(self, **kwargs):
        self.id = None
        self.ni = None # node index
        self.li = None # leaf index
        self.isroot = False
        self.isleaf = False
        self.label = None
        self.length = None
        self.support = None
        self.age = None
        self.parent = None
        self.children = []
        self.nchildren = 0
        self.left = None
        self.right = None
        self.treename = ""
        self.comment = ""
        self.apeidx = None
        self.meta = defaultdict(lambda:None)
        self.meta["cached"]=False
        ## self.length_comment = ""
        ## self.label_comment = ""
        if kwargs:
            for k, v in list(kwargs.items()):
                setattr(self, k, v)
        if self.id is None: self.id = id(self)

    def __copy__(self):
        return self.copy()

    def __repr__(self):
        v = []
        if self.isroot:
            v.append("root")
        elif self.isleaf:
            v.append("leaf")

        if self.label:
            v.append("'%s'" % self.label)

        s = ", ".join(v)

        nid = ((self.id if (self.id is not None) else self.ni) or
               '<%s>' % id(self))
        if s:
            s = "Node(%s, %s)" % (nid, s)
        else:
            s = "Node(%s)" % nid
        return s


    def __contains__(self, other):
        """
        For use with `in` keyword

        Args:
            other: Another node or node label.
        Returns:
            bool: Whether or not the other node is a descendant of self
        """
        otype = type(other)
        if other and otype in StringTypes:
            for x in self:
                if other == x.label:
                    return True
            return False
        else:
            assert otype == type(self)
            for x in self.iternodes():
                if other == x:
                    return True
        return False

    def __iter__(self):
        if not self.meta["cached"]:
            for node in self.iternodes():
                yield node
        else:
            for node in self.iternodes_cached():
                yield node
        # for node in self.iternodes():
        #     yield node

    def __len__(self):
        """
        Number of nodes descended from self

        Returns:
            int: Number of nodes descended from self (including self)
        """
        i = 0
        for n in self:
            i += 1
        return i

    def __bool__(self):
        return True

    def __getitem__(self, x):
        """
        Args:
            x: A Node, Node.id (int) or a Node.label (string)

        Returns:
            Node: Found node(s)

        """
        if self.meta["cached"] and type(x) == int:
            try:
                return self.iternodes_cached()[x] # Fast indexing using ni
            except IndexError:
                pass
        for n in self:
            if n==x or n.id==x or n.ni == x or (n.label and n.label==x):
                return n
        raise IndexError(str(x))

    def reindex(node, n=0, d=0, ni=0, li=0, ii=0,pi=0):
        """
        Iteratively attach 'ni',
        'ii', 'pi', and 'li' attributes to nodes
        """
        ni = 0
        ii = 0
        li = 0
        for n in node.iternodes():
            n.ni = ni
            ni += 1
            if n.isleaf:
                n.li = li
                li += 1
            else:
                n.ii = ii
                ii += 1
        for n in node.postiter():
            n.pi = pi
            pi += 1

    def ape_node_idx(self): # For use in phylorate plot
        i = 1
        for lf in self.leaves():
            lf.apeidx = i
            i += 1
        for n in [self]+self.clades():
            n.apeidx = i
            i += 1

    def get_root(self):
        if not self.isroot:
            return self.parent.get_root()
        else:
            return self

    def ascii(self, *args, **kwargs):
        """
        Create ascii tree.

        Keyword Args:
            unitlen (float): How long each unit should be rendered as.
              Defaults to 3.
            minwidth (float): Minimum width of the plot. Defaults to 50
            maxwidth (float): Maximum width of the plot. Defaults to None
            scaled (bool): Whether or not the tree is scaled. Defaults to False
            show_internal_labels (bool): Whether or not to show labels
              on internal nodes. Defaults to True.
        Returns:
            str: Ascii tree to be shown with print().
        """
        from .ascii import render
        return render(self, *args, **kwargs)

    def collapse(self, add=False, reindex=True):
        """
        Mutate function

        Remove self and collapse children to polytomy

        Args:
            add (bool): Whether or not to add self's length to children's
              length.
            reindex (bool): Whether to recalculate index attributes after mutating

        Returns:
            Node: Parent of self

        """
        assert self.parent
        p = self.prune(reindex=reindex)
        for c in self.children:
            p.add_child(c, reindex=reindex)
            if add and (c.length is not None):
                c.length += self.length
        self.children = []
        if reindex:
            p.get_root().set_iternode_cache()
            p.reindex()
        return p

    def copy(self, recurse=True, _par=None):
        """
        Return a shallow copy of self. If recurse = False, do not copy children,
        parents, or any attribute that is Node.

        Args:
            recurse (bool): Whether or not to copy children as well as self.

        Returns:
            Node: A copy of self.

        """
        for n in self.iternodes():
            self.cached = False
        newnode = Node()
        for attr, value in list(self.__dict__.items()):
            if (attr not in ("children", "parent") and
                not isinstance(value, Node)):
                setattr(newnode, attr, _copy(value))
        if recurse:
            newnode.children = [
                child.copy(True, _par = newnode) for child in self.children
                ]
            if _par:
                newnode.parent = _par
        for n in self.iternodes():
            self.cached = True
        newnode.set_iternode_cache()
        return newnode

    def leafsets(self, d=None, labels=False):
        """return a mapping of nodes to leaf sets (nodes or labels)"""
        d = d or {}
        if not self.isleaf:
            s = set()
            for child in self.children:
                if child.isleaf:
                    if labels:
                        s.add(child.label)
                    else:
                        s.add(child)
                else:
                    d = child.leafsets(d, labels)
                    s = s | d[child]
            d[self] = frozenset(s)
        return d

    def mrca(self, *nodes):
        """
        Find most recent common ancestor of *nodes*

        Args:
            *nodes (Node): Node objects
        Returns:
            Node: The MRCA of *nodes*
        """
        if len(nodes) == 1:
            nodes = tuple(nodes[0])
        if len(nodes) == 1:
            return nodes[0]
        nodes = set([ self[n] for n in nodes ])
        anc = []
        def f(n):
            seen = set()
            for c in n.children: seen.update(f(c))
            if n in nodes: seen.add(n)
            if seen == nodes and (not anc): anc.append(n)
            return seen
        f(self)
        return anc[0]

    def ismono(self, *leaves):
        """
        Test if leaf descendants are monophyletic

        Args:
            *leaves (Node): At least two leaf Node objects or labels

        Returns:
            bool: Are the leaf descendants monophyletic?

        """

        if len(leaves) == 1:
            # The only valid input of length 1 is a list of leaves.
            assert type(leaves[0]) == list, "Need more than one leaf for ismono(), got %s" % leaves

            leaves = leaves[0]
        assert all([ self[n].isleaf for n in leaves ]), "All given nodes must be leaves"

        assert len(leaves) > 1, (
            "Need more than one leaf for ismono(), got %s" % leaves
            )
        anc = self.mrca(leaves)
        if anc:
            return bool(len(anc.leaves())==len(leaves))

    def order_subtrees_by_size(self, n2s=None, recurse=False, reverse=False):
        """
        Order interal clades by size

        """
        if n2s is None:
            n2s = clade_sizes(self)
        if not self.isleaf:
            v = [ (n2s[c], c.label, c) for c in self.children ]
            v.sort()
            if reverse:
                v.reverse()
            self.children = [ x[-1] for x in v ]
            if recurse:
                for c in self.children:
                    c.order_subtrees_by_size(n2s, recurse=True, reverse=reverse)

    def ladderize(self, reverse=False, reindex=True):
        """
        Mutate function
        Rotate nodes so tree is ordered by clade size.

        WARNING: May cause strange results with functions that rely on
          pre- and post- ordering of nodes
        """
        assert self.isroot, "Must provide root node to ladderize tree"
        self.order_subtrees_by_size(recurse=True, reverse=reverse)
        return self

    def add_child(self, child, reindex=True):
        """
        Mutate function
        Add child as child of self

        Args:
            child (Node): A node object
            reindex (bool): Whether to recalculate index attributes after mutating

        """
        assert child not in self.children
        self.children.append(child)
        child.parent = self
        child.isroot = False
        self.nchildren += 1
        if reindex:
            self.get_root().set_iternode_cache()
            self.reindex()

    def bisect_branch(self, distance = 0.5, reindex=True):
        """
        Mutate function.
        Add new node as parent to self in the middle of branch to parent.

        Args:
            distance (float): What percentage along branch to place
              new node. Defaults to 0.5 (bisection). Higher numbers
              set the new node closer to the parent, lower
              numbers set it closer to child.
            reindex (bool): Whether to recalculate index attributes after mutating

        Returns:
            Node: A new node.
        """
        assert self.parent
        assert 0 < distance < 1
        parent = self.prune(reindex=reindex)
        n = Node()
        if self.length:
            n.length = self.length * (1-distance)
            self.length *= distance
        parent.add_child(n, reindex=reindex)
        n.add_child(self, reindex=reindex)
        if reindex:
            self.get_root().set_iternode_cache()
            self.reindex()

        return n

    def remove_child(self, child, reindex=True):
        """
        Mutate function.
        Remove child from self.

        Args:
            child (Node): A node object that is a child of self
            reindex (bool): Whether to recalculate index attributes after mutating

        """
        assert child in self.children, "node '%s' not child of node '%s'" % (child.label or child.id, self.label or self.id)
        self.children.remove(child)
        child.parent = None
        self.nchildren -= 1
        if not self.children:
            self.isleaf = True
        if reindex:
            self.get_root().set_iternode_cache()
            self.reindex()

    def labeled(self):
        """
        Return a list of all descendant nodes that are labeled

        Returns:
            list: All descendants of self that are labeled (including self)
        """
        return [ n for n in self if n.label ]

    def leaves(self, f=None):
        """
        Return a list of leaves. Can be filtered with f.

        Args:
            f (function): A function that evaluates to True if called with
              desired node as the first input

        Returns:
            list: A list of leaves that are true for f (if f is given)
        """
        if f:
            return [ n for n in self if (n.isleaf and f(n)) ]
        return [ n for n in self if n.isleaf ]

    def internals(self, f=None):
        """
        Return a list nodes that have children (internal nodes)

        Args:
            f (function): A function that evaluates to true if called with
              desired node as the first input

        Returns:
            list: A list of internal nodes that are true for f (if f is given)
        """
        if f:
            return [ n for n in self if (n.children and f(n)) ]
        return [ n for n in self if n.children ]

    def clades(self):
        """
        Get internal nodes descended from self

        Returns:
            list: A list of internal nodes descended from (and not including) self.
        """
        return [ n for n in self if (n is not self) and not n.isleaf ]

    def iternodes(self,f=None):
        """
        List of nodes descendant from self - including self
        Yields:
            Node: Nodes descended from self (including self) in
              preorder sequence
        """
        s = []
        s.append(self)
        n = self
        if f is None:
            f = lambda x: True
        while len(s) != 0:
            n = s.pop()
            if f(n):
                yield n
            for child in reversed(n.children):
                s.append(child)
    def set_iternode_cache(self):
        """
        Store iteration order for faster access.
        """
        for n in self.iternodes():
            n.meta["iterlist"] = list(n.iternodes())

    def iternodes_cached(self, f=None, force=False):
        """
        Cached version of iternodes. Faster, but requires that the tree
        is static and not being changed.
        """
        if f is None:
            f = lambda x: True
        if (self.meta["iterlist"] is None) or force:
            self.set_iternode_cache()
        return [n for n in self.meta["iterlist"] if f(n)]
    def iterleaves(self):
        """
        Yield leaves descendant from self
        """
        return self.iternodes(lambda x:x.isleaf)

    def preiter(self, f=None):
        """
        Yield nodes in preorder sequence
        """
        for n in self.iternodes(f=f):
            yield n

    def postiter(self, f=None):
        """
        Yield nodes in postorder sequence
        """
        if not self.isleaf:
            for child in self.children:
                for n in child.postiter():
                    if (f and f(n)) or (not f):
                        yield n
        if (f and f(self)) or (not f):
            yield self

    def descendants(self, order="pre", v=None, f=None):
        """
        Return a list of nodes descendant from self - but _not_
        including self!

        Args:
            order (str): Indicates wether to return nodes in preorder or
              postorder sequence. Optional, defaults to "pre"
            f (function): filtering function that evaluates to True if desired
              node is called as the first parameter.

        Returns:
            list: A list of nodes descended from self not including self.
        """
        v = v or []
        for child in self.children:
            if (f and f(child)) or (not f):
                if order == "pre":
                    v.append(child)
                else:
                    v.insert(0, child)
            if child.children:
                child.descendants(order, v, f)
        return v

    def drop_tip(self, nodes):
        """
        Return a NEW TREE with the given tips dropped from it. Does not
        affect old tree.

        Args:
            nodes (list): Leaf nodes or labels of leaf nodes
        Returns:
            Node (Node): New root node with tips dropped
        """
        t = self.copy()

        for n in t.iternodes():
            n.meta["cached"]=False

        nodes = [ t[self[x].id] for x in nodes ]
        nodes = sorted(nodes, key=lambda x: x.ni)
        assert all([ x.isleaf for x in nodes ]), "All nodes given must be tips"
        root = t

        for node in nodes:
            cp = node.parent # current parent
            cp.remove_child(node, reindex=False)
            if cp.length:
                node.length += cp.length
            if len(cp.children) == 1: # If parent is now a "knee"...
                try:
                    cp.excise(reindex=False) # Remove parent
                except AssertionError: # If parent was the root, assign new root
                    t.isroot = False
                    root = cp.children[0]
                    root.parent = None
            elif len(cp.children) == 0:
                cpp = cp.parent # current parent's parent
                cp.parent.remove_child(cp, reindex=False)
                if cpp.nchildren == 1:
                    try:
                        cpp.excise()
                    except AssertionError:
                        t.isroot = False
                        root = cpp.children[0]
                        root.parent = None
                        root.isroot = True
        for n in root.descendants():
            # This removes all knees in the tree. It mimics what ape's
            # drop.tip function does. Unsure what the behavior for ivy
            # should be.
            if n.nchildren == 1:
                n.excise(reindex=False)
        root.isroot = True
        root.set_iternode_cache()
        for n in root:
            n.meta["cached"] = True
        return root

    def keep_tip(self, nodes):
        """
        Return a NEW TREE containing only the given tips.

        Args:
            nodes (list): Leaf nodes or labels of leaf notes
        Returns:
            Node (Node): New root node containing only given tips
        """
        nodes = [ self[x] for x in nodes ]
        assert all([ x.isleaf for x in nodes ]), "All nodes given must be tips"
        to_drop = [ l for l in self.leaves() if not l in nodes ]
        return self.drop_tip(to_drop)

    def get(self, f, *args, **kwargs):
        """
        Return the first node found by node.find()

        Args:
            f (function): A function that evaluates to True if desired
              node is called as the first parameter.
            *args, **kwargs: Additional args called by f
        Returns:
            Node: The first node found by node.find()

        """
        v = self.find(f, *args, **kwargs)
        try:
            return next(v)
        except StopIteration:
            return None

    def grep(self, s, ignorecase=True):
        """
        Find nodes by regular-expression search of labels

        Args:
            s (str): String to search.
            ignorecase (bool): Indicates to ignore case. Defaults to true.

        Returns:
            lsit: A list of node objects whose labels were matched by s.

        """
        import re
        if ignorecase:
            pattern = re.compile(s, re.IGNORECASE)
        else:
            pattern = re.compile(s)

        search = pattern.search
        return [ x for x in self if x.label and search(x.label) ]

    def lgrep(self, s, ignorecase=True):
        """
        Find leaves by regular-expression search of labels

        Args:
            s (str): String to search.
            ignorecase (bool): Indicates to ignore case. Defaults to true.

        Returns:
            lsit: A list of node objects whose labels were matched by s.

        """
        return [ x for x in self.grep(s, ignorecase=ignorecase) if x.isleaf ]

    def bgrep(self, s, ignorecase=True):
        """
        Find branches (internal nodes) by regular-expression search of
        labels

        Args:
            s (str): String to search.
            ignorecase (bool): Indicates to ignore case. Defaults to true.

        Returns:
            lsit: A list of node objects whose labels were matched by s.

        """
        return [ x for x in self.grep(s, ignorecase=ignorecase) if
               (not x.isleaf) ]

    def find(self, f, *args, **kwargs):
        """
        Find descendant nodes (generator version)

        Args:
            f: Function or a string.  If a string, it is converted to a
              function for finding *f* as a substring in node labels.
              Otherwise, *f* should evaluate to True if called with a desired
              node as the first parameter.

        Yields:
            Node: Found nodes in preorder sequence.

        """
        if not f: return
        if type(f) in StringTypes:
            func = lambda x: (f or None) in (x.label or "")
        else:
            func = f
        for n in self.iternodes():
            if func(n, *args, **kwargs):
                yield n

    def findall(self, f, *args, **kwargs):
        """
        Find descendant nodes (list version)

        Args:
            f: Function or a string.  If a string, it is converted to a
              function for finding *f* as a substring in node labels.
              Otherwise, *f* should evaluate to True if called with a desired
              node as the first parameter.

        Yields:
            Node: Found nodes in preorder sequence.

        """
        return list(self.find(f, *args, **kwargs))

    def is_same_tree(self, tree):
        """
        Test if two trees are the same (same topology, characteristics, labels,
        etc.) Ignores IDs by default.

        Args:
            tree (Node): Another tree to compare to
            verbose (bool): Whether or not to print a message containing
              the non-matching properties
        Returns:
            bool: Whether or not the trees are the same.
        """
        assert self.isroot and tree.isroot, "Must compare root nodes"
        # Recursively check properties of both trees EXCEPT for IDs and children/parents
        # (IDs are ignored by default and children/parents will be checked
        # during the enumeration of all nodes)
        a_tree = self.copy()
        b_tree = tree.copy()

        return a_tree._is_isomorphic(b_tree)

    def _is_isomorphic(self, node):
        """
        Two nodes are isomorphic if all of their properties are identical;
        Two leaves with identical properties are isomorphic
        Two internals are identical if their child nodes are identical
        in any order.

        """

        propsToCheck = ["label","nchildren"]
        numericPropsToCheck = ["length"]

        for prop in propsToCheck:
            if getattr(self, prop) != getattr(node, prop):
                return False
        for prop in numericPropsToCheck:
            att1 = getattr(self,prop);att2 = getattr(node,prop)
            if att1 and att2:
                if not np.isclose(att1, att2):
                    return False
        if self.nchildren == 0 and node.nchildren == 0:
            return True
        else:
            for s in itertools.permutations(self.children):
                if all([ n._is_isomorphic(node.children[i]) for i,n in enumerate(s) ]):
                    return True

        return False

    def prune(self, reindex=True):
        """
        Mutate function
        Remove self if self is not root.

        All descendants of self are also removed

        Args:
            reindex (bool): Whether to recalculate index attributes after mutating tree.

        Returns:
            Node: Parent of self. If parent had only two children,
              parent is now a 'knee' and can be removed with excise.

        """
        p = self.parent
        if p:
            p.remove_child(self, reindex=reindex)
        if reindex:
            p.get_root().set_iternode_cache()
            p.reindex()
        return p

    def excise(self, reindex=True):
        """
        Mutate function
        For 'knees': remove self from between parent and single child

        Args:
            reindex (bool): Whether to recalculate index attributes after mutating tree.

        """
        assert self.parent
        assert len(self.children)==1
        p = self.parent
        c = self.children[0]
        if c.length is not None and self.length is not None:
            c.length += self.length
        c.prune(reindex=reindex)
        self.prune(reindex=reindex)
        p.add_child(c,reindex=reindex)
        if reindex:
            p.get_root().set_iternode_cache()
            p.reindex()
        return p

    def graft(self, node, reindex=True):
        """
        Mutate function
        Add node as sister to self.
        Args:
            node (Node): Node to graft to tree
            reindex (bool): Whether to recalculate index attributes after mutating tree.

        """
        parent = self.parent
        parent.remove_child(self, reindex=reindex)
        n = Node()
        n.add_child(self, reindex=reindex)
        n.add_child(node, reindex=reindex)
        parent.add_child(n, reindex=reindex)
        if reindex:
            self.get_root().set_iternode_cache()
            self.reindex()

    def leaf_distances(self, measure="length"):
        """
        RR: I don't quite understand the structure of the output.
        """
        from collections import defaultdict
        store = defaultdict(lambda:defaultdict(lambda:0))
        nodes = [ x for x in self if x.children ]
        for lf in self.leaves():
            x = lf.length
            for n in lf.rootpath(self):
                store[n][lf] = x
                x += (n.length or 0)
        return store

    def rootpath(self, end=None, stop=None):
        """
        Iterate over parent nodes toward the root, or node *end* if
        encountered.

        Args:
            end (Node): A Node object to iterate to (instead of iterating
              towards root). Optional, defaults to None
            stop (function): A function that returns True if desired node is
              called as the first parameter. Optional, defaults to None

        Yields:
            Node: Nodes in path to root (or end).

        """
        n = self.parent
        while 1:
            if n is None: raise StopIteration
            yield n
            if n.isroot or (end and n == end) or (stop and stop(n)):
                raise StopIteration
            n = n.parent

    def rootpath_length(self, end=None):
        """
        Get length from self to root(if end is None) or length
        from self to an ancestor node (if end is an ancestor to self)

        Args:
            end (Node): A node object

        Returns:
            float: The length from self to root/end

        """
        n = self
        x = 0.0
        while n.parent:
            x += n.length
            if n.parent == end:
                break
            n = n.parent
        return x
        ## f = lambda x:x.parent==end
        ## v = [self.length]+[ x.length for x in self.rootpath(stop=f)
        ##                     if x.parent ]
        ## assert None not in v
        ## return sum(v)

    def max_tippath(self, first=True):
        """
        Get the maximum length from self to a leaf node
        """
        v = 0
        if self.children:
            v = max([ c.max_tippath(False) for c in self.children ])
        if not first:
            if self.length is None: v += 1
            else: v += self.length
        return v

    def subtree_mapping(self, labels, clean=False):
        """
        Find the set of nodes in 'labels', and create a new tree
        representing the subtree connecting them.  Nodes are assumed
        to be non-nested.

        Returns:
            dict: a mapping of old nodes to new nodes and vice versa.

        TODO: test this, high bug probability
        """
        d = {}
        oldtips = [ x for x in self.leaves() if x.label in labels ]
        for tip in oldtips:
            path = list(tip.rootpath())
            for node in path:
                if node not in d:
                    newnode = Node()
                    newnode.isleaf = node.isleaf
                    newnode.length = node.length
                    newnode.label = node.label
                    d[node] = newnode
                    d[newnode] = node
                else:
                    newnode = d[node]

                for child in node.children:
                    if child in d:
                        newchild = d[child]
                        if newchild not in newnode.children:
                            newnode.add_child(newchild)
        d["oldroot"] = self
        d["newroot"] = d[self]
        if clean:
            n = d["newroot"]
            while 1:
                if n.nchildren == 1:
                    oldnode = d[n]
                    del d[oldnode]; del d[n]
                    child = n.children[0]
                    child.parent = None
                    child.isroot = True
                    d["newroot"] = child
                    d["oldroot"] = d[child]
                    n = child
                else:
                    break

            for tip in oldtips:
                newnode = d[tip]
                while 1:
                    newnode = newnode.parent
                    oldnode = d[newnode]
                    if newnode.nchildren == 1:
                        child = newnode.children[0]
                        if newnode.length:
                            child.length += newnode.length
                        newnode.remove_child(child)
                        if newnode.parent:
                            parent = newnode.parent
                            parent.remove_child(newnode)
                            parent.add_child(child)
                        del d[oldnode]; del d[newnode]
                    if not newnode.parent:
                        break

        return d

    def reroot(self, newroot, distance = 0.5):
        """
        Reroot the tree between newroot and its parent.
        By default, the new node is halfway in between
        newroot and its current parent. Works by unrooting the tree, then
        rerooting it at the new node.

        Returns a NEW tree. Does not affect old tree

        Args:
            newroot: Node or str of node label. Cannot be child of
              current root.
            distance (float): What percentage along branch to place
              new node. Defaults to 0.5 (bisection). Higher numbers
              set the new node closer to the parent, lower
              numbers set it closer to child.
        Returns:
            Node: Root node of new rerooted tree.
        """
        oldroot = self.copy()
        for n in oldroot.iternodes():
            n.meta["cached"]=False
        newroot = self[newroot]
        oldroot.isroot = False
        newroot = oldroot[newroot.id]
        assert newroot in oldroot
        assert newroot not in oldroot.children
        newtree = newroot.bisect_branch(distance, reindex=False)

        root_path = list(newtree.rootpath())
        newtree.parent = None
        newtree.children.append(root_path[0])

        newparent = newtree
        newlen = newtree.length
        for node in root_path:
            node.children = [ x for x in node.children if x is not newparent ]
            node.children.append(node.parent)
            node.parent = newparent
            oldlen = node.length
            node.length = newlen
            newlen = oldlen
            newparent = node
        root_path[-1].children = [ x for x in root_path[-1].children if x ]
        try:
            root_path[-1].excise(reindex=False)
        except:
            pass

        newtree.isroot = True
        newtree.set_iternode_cache()
        for n in newroot.iternodes():
            n.meta["cached"]=False
        newtree.reindex()
        return newtree
    def makeroot(self, shift_labels=False):
        """
        shift_labels: flag to shift internal parent-child node labels
        when internode polarity changes; suitable e.g. if internal node
        labels indicate unrooted bipartition support
        """
        v = list(self.rootpath())
        v[-1].isroot = False
        v.reverse()
        for node in v[1:] + [self]:
            # node is current node; cp is current parent
            cp = node.parent
            cp.remove_child(node)
            node.add_child(cp)
            cp.length = node.length
            if shift_labels:
                cp.label = node.label
        self.isroot = True
        return self

    def write(self, outfile=None, format="newick", length_fmt=":%g", end=True,
              clobber=False):
        if format=="newick":
            s = write_newick(self, outfile, length_fmt, True, clobber)
            if not outfile:
                return s

    def get_siblings(self):
        """
        Return list of siblings of node
        """
        assert self.parent is not None
        return [c for c in self.parent.children if not c==self]
    def ntaxa(self):
        """
        Number of leaves descended from self
        """
        return len(self.leaves())
    def tiplabels(self):
        """
        List of labels of leaves descended from self, in preorder sequence
        """
        return [n.label for n in self.leaves()]



reroot = Node.reroot

def remove_singletons(root, add=True):
    "Remove descendant nodes that are the sole child of their parent"
    for leaf in root.leaves():
        for n in leaf.rootpath():
            if n.parent and len(n.parent.children)==1:
                n.collapse(add)

def cls(root):
    """
    Get clade sizes of whole tree
    Args:
        * root: A root node

    Returns:
        * A dict mapping nodes to clade sizes

    """
    results = {}
    for node in root.postiter():
        if node.isleaf:
            results[node] = 1
        else:
            results[node] = sum(results[child] for child in node.children)
    return results

def clade_sizes(node, results={}):
    """Map node and descendants to number of descendant tips"""
    size = int(node.isleaf)
    if not node.isleaf:
        for child in node.children:
            clade_sizes(child, results)
            size += results[child]
    results[node] = size
    return results

def write(node, outfile=None, format="newick", length_fmt=":%g",
          clobber=False):
    if format=="newick" or ((type(outfile) in StringTypes) and
                            (outfile.endswith(".newick") or
                             outfile.endswith(".new"))):
        s = write_newick(node, outfile, length_fmt, True, clobber)
        if not outfile:
            return s

def write_newick(node, outfile=None, length_fmt=":%g", end=False,
                 clobber=False):
    if not node.isleaf:
        node_str = "(%s)%s" % \
                   (",".join([ write_newick(child, outfile, length_fmt,
                                            False, clobber)
                               for child in node.children ]),
                    (node.label or "")
                    )
    else:
        node_str = "%s" % node.label

    if node.length is not None:
        length_str = length_fmt % node.length
    else:
        length_str = ""

    semicolon = ""
    if end:
        semicolon = ";"
    s = "%s%s%s" % (node_str, length_str, semicolon)
    if end and outfile:
        flag = False
        if type(outfile) in StringTypes:
            if not clobber:
                assert not os.path.isfile(outfile), "File '%s' exists! (Set clobber=True to overwrite)" % outfile
            flag = True
            outfile = open(outfile, "w")
        outfile.write(s)
        try:
            outfile.close()
        except:
            pass

    return s

def read(data, format=None, treename=None, ttable=None):
    """
    Read a single tree from *data*, which can be a Newick string, a
    file name, or a file-like object with `tell` and 'read`
    methods. *treename* is an optional string that will be attached to
    all created nodes.

    Args:
        data: A file or file-like object or newick string

    Returns:
        Node: The root node.
    """
    from . import newick


    def strip(s):
        fname = os.path.split(s)[-1]
        head, tail = os.path.splitext(fname)
        tail = tail.lower()
        if tail in (".nwk", ".tre", ".tree", ".newick", ".nex"):
            return head
        else:
            return fname

    if (not format):
        if (type(data) in StringTypes) and os.path.isfile(data):
            s = data.lower()
            for tail in ".nex", ".nexus", ".tre":
                if s.endswith(tail):
                    format="nexus"
                    break

    if (not format):
        format = "newick"

    if format == "newick":
        if type(data) in StringTypes:
            if os.path.isfile(data):
                treename = strip(data)
                with open(data, "r") as f:
                    parsed = newick.parse(f, treename=treename,
                                        ttable=ttable)
                out = parsed
            else:
                out = newick.parse(data, ttable=ttable)

        elif (hasattr(data, "tell") and hasattr(data, "read")):
            treename = strip(getattr(data, "name", None))
            out = newick.parse(data, treename=treename, ttable=ttable)
    elif format == "nexus-dendropy":
        import dendropy
        if type(data) in StringTypes:
            if os.path.isfile(data):
                treename = strip(data)
                out = newick.parse(
                    str(dendropy.Tree.get_from_path(data, "nexus")),
                    treename=treename
                    )
            else:
                out = newick.parse(
                    str(dendropy.Tree.get_from_string(data, "nexus"))
                    )

        elif (hasattr(data, "tell") and hasattr(data, "read")):
            treename = strip(getattr(data, "name", None))
            out = newick.parse(
                str(dendropy.Tree.get_from_stream(data, "nexus")),
                treename=treename
                )
        else:
            pass

    elif format == "nexus":
        if type(data) in StringTypes:
            if os.path.isfile(data):
                with open(data) as infile:
                    rec = next(newick.nexus_iter(infile))
                    if rec:
                        out = rec.parse()
            else:
                rec = next(newick.nexus_iter(StringIO(data)))
                if rec:
                    out = rec.parse()
        else:
            rec = next(newick.nexus_iter(data))
            if rec:
                out = rec.parse()
    else:
        # implement other tree formats here (nexus, nexml etc.)
        raise IOError("format '%s' not implemented yet" % format)
    try:
        out.set_iternode_cache()
    except UnboundLocalError:
        raise IOError("unable to read tree from '%s'" % data)

    for n in out:
        n.meta["cached"] = True
    return out

def readmany(data, format="newick"):
    """Iterate over trees from a source."""
    if type(data) in StringTypes:
        if os.path.isfile(data):
            data = open(data)
        else:
            data = StringIO(data)

    if format == "newick":
        for line in data:
            yield newick.parse(line)
    elif format == "nexus":
        for rec in newick.nexus_iter(data):
            yield rec.parse()
    else:
        raise Exception("format '%s' not recognized" % format)
    data.close()

## def randomly_resolve(n):
##     assert len(n.children)>2

## def leaf_mrcas(root):
##     from itertools import product, izip, tee
##     from collections import OrderedDict
##     from numpy import empty
##     mrca = OrderedDict()
##     def pairwise(iterable, tee=tee, izip=izip):
##         a, b = tee(iterable)
##         next(b, None)
##         return izip(a, b)
##     def f(n):
##         if n.isleaf:
##             od = OrderedDict(); od[n] = n.length
##             return od
##         d = [ f(c) for c in n.children ]
##         for i, j in pairwise(xrange(len(d))):
##             di = d[i]; dj =d[j]
##             for ni, niv in di.iteritems():
##                 for nj, njv in dj.iteritems():
##                     mrca[(ni,nj)] = n
##             d[j].update(di)
##         return d[j]
##     f(root)
##     return mrca

def C(leaves, internals):
    from scipy.sparse import lil_matrix
    m = lil_matrix((len(internals), len(leaves)))
    for lf in leaves:
        v = lf.length if lf.length is not None else 1
        for n in lf.rootpath():
            m[n.ii,lf.li] = v
            v += n.length if n.length is not None else 1
    return m.tocsc()

def load_chars(filename, colNames=True, rowNames=False):
    """
    Given a filename pointing to a CSV with species names as column one
    and characters as remaining columns, return a dictionary mapping
    names to characters
    """
    chars = {}
    with open(filename, "r") as f:
        read = csv.reader(f, delimiter=",", quotechar='"')
        for i,row in enumerate(read):
            if colNames and i==0:
                charNames = row[1:]
            elif i==0:
                charNames = [ "c"+i for i in range(len(row[1:])) ]
                chars[row[0]] = { char:row[j+1] for j, char in enumerate(charNames) }
            else:
                chars[row[0]] = { char:row[j+1] for j, char in enumerate(charNames) }
    return chars
