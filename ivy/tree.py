"""
The Node class and functions for creating trees from Newick strings,
etc.

ivy does not have a Tree class per se, as most functions operate
directly on Node objects.
"""
import os, types
from storage import Storage
from copy import copy as _copy
from matrix import vcv

## class Tree(object):
##     """
##     A simple Tree class.
##     """
##     def __init__(self, data=None, format="newick", name=None, ttable=None):
##         self.root = None
##         if data:
##             self.root = read(data, format, name, ttable)
##         self.name = name
##         self.ttable = ttable

##     def __getattribute__(self, a):
##         r = object.__getattribute__(self, 'root')
##         try:
##             return object.__getattribute__(r, a)
##         except AttributeError:
##             return object.__getattribute__(self, a)
        

class Node(object):
    """
    A basic Node class with attributes and references to child nodes
    ('children', a list) and 'parent'.
    """
    def __init__(self):
        self.id = None
        self.isroot = False
        self.isleaf = False
        self.label = None
        self.length = None
        self.support = None
        self.age = None
        self.parent = None
        self.children = []
        self.nchildren = 0
        self.treename = None

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

        if s:
            s = "Node(%s, %s)" % (self.id, s)
        else:
            s = "Node(%s)" % self.id
        return s

    def __contains__(self, other):
        otype = type(other)
        if other and otype in types.StringTypes:
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
        for node in self.iternodes():
            yield node

    def __len__(self):
        i = 0
        for n in self:
            i += 1
        return i
            
    def __getitem__(self, x):
        """
        x is a Node, Node.id (int) or a Node.label (string)
        """
        for n in self:
            if n==x or n.id==x or (n.label and n.label==x):
                return n
        raise IndexError(str(x))

    def ascii(self, *args, **kwargs):
        from ascii import render
        return render(self, *args, **kwargs)

    def collapse(self, add=False):
        assert self.parent
        p = self.prune()
        for c in self.children:
            p.add_child(c)
            if add and (c.length is not None):
                c.length += self.length
        self.children = []
        return p

    def copy(self, recurse=False):
        """
        Return a copy of the node, but not copies of children, parent,
        or any attribute that is a Node.
        
        If `recurse` is True, recursively copy child nodes.

        TODO: test this function.
        """
        newnode = Node()
        for attr, value in self.__dict__.items():
            if (attr not in ("children", "parent") and
                not isinstance(value, Node)):
                setattr(newnode, attr, _copy(value))
            if recurse:
                newnode.children = [
                    child.copy(True) for child in self.children
                    ]
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
        """
        if len(nodes) == 1:
            nodes = list(nodes[0])
        if len(nodes) == 1:
            return nodes[0]
        ## assert len(nodes) > 1, (
        ##     "Need more than one node for mrca(), got %s" % nodes
        ##     )
        def f(x):
            if isinstance(x, Node):
                return x
            elif type(x) in types.StringTypes:
                return self.find(x)
        nodes = map(f, nodes)
        assert all(filter(lambda x: isinstance(x, Node), nodes))

        v = [ list(n.rootpath()) for n in nodes if n in self ]
        if len(v) == 1:
            return v[0][0]
        anc = None
        while 1:
            s = set([ x.pop() for x in v if x ])
            if len(s) == 1:
                anc = list(s)[0]
            else:
                break
        return anc

    def ismono(self, *leaves):
        "Test if leaf descendants are monophyletic"
        if len(leaves) == 1:
            leaves = list(leaves)[0]
        assert len(leaves) > 1, (
            "Need more than one leaf for ismono(), got %s" % leaves
            )
        anc = self.mrca(leaves)
        if anc:
            return bool(len(anc.leaves())==len(leaves))
        
    def order_subtrees_by_size(self, n2s=None, recurse=False, reverse=False):
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

    def ladderize(self, reverse=False):
        self.order_subtrees_by_size(recurse=True, reverse=reverse)

    def add_child(self, child):
        assert child not in self.children
        self.children.append(child)
        child.parent = self
        self.nchildren += 1

    def bisect_branch(self):
        assert self.parent
        parent = self.prune()
        n = Node()
        if self.length:
            n.length = self.length/2.0
            self.length /= 2.0
        parent.add_child(n)
        n.add_child(self)
        return n

    def remove_child(self, child):
        assert child in self.children
        self.children.remove(child)
        child.parent = None
        self.nchildren -= 1
        if not self.children:
            self.isleaf = True

    def labeled(self):
        return [ n for n in self if n.label ]

    def leaves(self):
        return [ n for n in self if n.isleaf ]

    def iternodes(self, f=None):
        """
        generate a list of nodes descendant from self - including self
        """
        if (f and f(self)) or (not f):
            yield self
        if not self.isleaf:
            for child in self.children:
                for n in child.iternodes():
                    if (f and f(n)) or (not f):
                        yield n

    def preiter(self):
        for n in self.iternodes():
            yield n

    def postiter(self):
        if not self.isleaf:
            for child in self.children:
                for n in child.postiter():
                    yield n
        yield self

    def descendants(self, order="pre", v=None, f=None):
        """
        Return a list of nodes descendant from self - but _not_
        including self!

        f = filtering function
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

    def get(self, f, *args, **kwargs):
        """
        Return the first node found by node.find()
        """
        return self.find(f, *args, **kwargs).next()

    def grep(self, s, ignorecase=True):
        """
        Find nodes by regular-expression search of labels
        """
        import re
        if ignorecase:
            pattern = re.compile(s, re.IGNORECASE)
        else:
            pattern = re.compile(s)

        search = pattern.search
        return [ x for x in self if x.label and search(x.label) ]

    def lgrep(self, s, ignorecase=True):
        "Find leaves by regular-expression search of labels"
        return [ x for x in self.grep(s) if x.isleaf ]

    def bgrep(self, s, ignorecase=True):
        """
        Find branches (internal nodes) by regular-expression search of
        labels
        """
        return [ x for x in self.grep(s) if (not x.isleaf) ]

    def find(self, f, *args, **kwargs):
        """
        Find descendant nodes. *f* can be a function or a string.  If
        a string, it is converted to a function for finding *f* as a
        substring in node labels.  Otherwise, *f* should evaluate to
        True if called with a desired node as the first parameter, and
        *args* and *kwargs* as additional unnamed and named
        parameters, respectively.

        Returns: a generator yielding found nodes in preorder sequence.
        """
        if not f: return
        if type(f) in types.StringTypes:
            func = lambda x: (f or None) in (x.label or "")
        else:
            func = f
        for n in self.iternodes():
            if func(n, *args, **kwargs):
                yield n

    def findall(self, f, *args, **kwargs):
        "Return a list of found nodes."
        return list(self.find(f, *args, **kwargs))

    def prune(self):
        p = self.parent
        if p:
            p.remove_child(self)
        return p

    def graft(self, node):
        parent = self.parent
        parent.remove_child(self)
        n = Node()
        n.add_child(self)
        n.add_child(node)
        parent.add_child(n)

    def leaf_distances(self, store=None):
        """
        for each internal node, calculate the distance to each leaf,
        measured in branch length or internodes
        """
        if store is None:
            store = {}
        leaf2len = {}
        if self.children:
            for child in self.children:
                dist = child.length
                child.leaf_distances(store)
                if child.isleaf:
                    leaf2len[child] = dist
                else:
                    for k, v in store[child].items():
                        leaf2len[k] = v + dist
        else:
            leaf2len[self] = {self: 0}
        store[self] = leaf2len
        return store

    def rootpath(self, end=None):
        """
        Iterate over parent nodes toward the root, or node *end* if
        encountered.
        """
        n = self
        while 1:
            yield n
            if n.isroot or (end and n == end): break
            if n.parent: n = n.parent
            else: break

    def subtree_mapping(self, labels, clean=False):
        """
        Find the set of nodes in 'labels', and create a new tree
        representing the subtree connecting them.  Nodes are assumed
        to be non-nested.

        Return: a mapping of old nodes to new nodes and vice versa.

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

    def reroot(self, newroot):
        assert newroot in self
        self.isroot = False
        newroot.isroot = True
        v = []
        n = newroot
        while 1:
            v.append(n)
            if not n.parent: break
            n = n.parent
        v.reverse()
        for i, cp in enumerate(v[:-1]):
            node = v[i+1]
            # node is current node; cp is current parent
            cp.remove_child(node)
            node.add_child(cp)
            cp.length = node.length
        return newroot

reroot = Node.reroot

def remove_singletons(root, add=True):
    "Remove descendant nodes that are the sole child of their parent"
    for leaf in root.leaves():
        for n in leaf.rootpath():
            if n.parent and len(n.parent.children)==1:
                n.collapse(add)

def cls(root):
    results = {}
    for node in root.postiter():
        if node.isleaf:
            results[node] = 1
        else:
            results[node] = sum(results[child] for child in root.children)
    return results

def clade_sizes(node, results={}):
    "Map node and descendants to number of descendant tips"
    size = int(node.isleaf)
    if not node.isleaf:
        for child in node.children:
            clade_sizes(child, results)
            size += results[child]
    results[node] = size
    return results

def write(node, outfile=None, format="newick", length_fmt=":%g"):
    if format=="newick":
        return write_newick(node, outfile, length_fmt, end=True)
    
def write_newick(node, outfile=None, length_fmt=":%g", end=True):
    if not node.isleaf:
        node_str = "(%s)%s" % \
                   (",".join([ write_newick(child, outfile, length_fmt, False)
                               for child in node.children ]),
                    node.label or ""
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
        if type(outfile) in types.StringTypes:
            assert (not os.path.isfile(outfile)), "File '%s' exists!"
            flag = True
            outfile = file(outfile, "w")
        outfile.write(s)
        if flag:
            outfile.close()
    return s
    
def read(data, format="newick", treename=None, ttable=None):
    """
    Read a single tree from *data*, which can be a Newick string, a
    file name, or a file-like object with `tell` and 'read`
    methods. *treename* is an optional string that will be attached to
    all created nodes.

    Returns: *root*, the root node.
    """
    import newick
    
    def strip(s):
        fname = os.path.split(s)[-1]
        head, tail = os.path.splitext(fname)
        if tail in (".nwk", ".tre", ".tree", ".newick"):
            return head
        else:
            return fname

    if format == "newick":
        if type(data) in types.StringTypes:
            if os.path.isfile(data):
                treename = strip(data)
                return newick.parse(file(data), treename=treename,
                                    ttable=ttable)
            else:
                return newick.parse(data, ttable=ttable)

        elif (hasattr(data, "tell") and hasattr(data, "read")):
            treename = strip(getattr(data, "name", None))
            return newick.parse(data, treename=treename, ttable=ttable)
    else:
        # implement other tree formats here (nexus, nexml etc.)
        raise IOError, "format '%s' not implemented yet" % format

    raise IOError, "unable to read tree from '%s'" % data

def readmulti(data, format="newick"):
    "Iterate over trees from a source."
    pass
