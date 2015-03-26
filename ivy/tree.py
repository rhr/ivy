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
import newick
from itertools import izip_longest

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
        self.length_comment = ""
        self.label_comment = ""
        if kwargs:
            for k, v in kwargs.items():
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

    def __nonzero__(self):
        return True
            
    def __getitem__(self, x):
        """
        x is a Node, Node.id (int) or a Node.label (string)
        """
        for n in self:
            if n==x or n.id==x or n.ni == x or (n.label and n.label==x):
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

    ## def mrca(self, *nodes):
    ##     """
    ##     Find most recent common ancestor of *nodes*
    ##     """
    ##     if len(nodes) == 1:
    ##         nodes = tuple(nodes[0])
    ##     if len(nodes) == 1:
    ##         return nodes[0]
    ##     ## assert len(nodes) > 1, (
    ##     ##     "Need more than one node for mrca(), got %s" % nodes
    ##     ##     )
    ##     def f(x):
    ##         if isinstance(x, Node):
    ##             return x
    ##         elif type(x) in types.StringTypes:
    ##             return self.find(x)
    ##     nodes = map(f, nodes)
    ##     assert all(filter(lambda x: isinstance(x, Node), nodes))

    ##     #v = [ list(n.rootpath()) for n in nodes if n in self ]
    ##     v = [ list(x) for x in izip_longest(*[ reversed(list(n.rootpath()))
    ##                                            for n in nodes if n in self ]) ]
    ##     if len(v) == 1:
    ##         return v[0][0]
    ##     anc = None
    ##     for x in v:
    ##         s = set(x)
    ##         if len(s) == 1: anc = list(s)[0]
    ##         else: break
    ##     return anc

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
        return self

    def add_child(self, child):
        assert child not in self.children
        self.children.append(child)
        child.parent = self
        child.isroot = False
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

    def leaves(self, f=None):
        if f: return [ n for n in self if (n.isleaf and f(n)) ]
        return [ n for n in self if n.isleaf ]

    def internals(self, f=None):
        if f: return [ n for n in self if (n.children and f(n)) ]
        return [ n for n in self if n.children ]

    def clades(self):
        return [ n for n in self if not n.isleaf ]

    def iternodes(self, f=None):
        """
        generate a list of nodes descendant from self - including self
        """
        if (f and f(self)) or (not f):
            yield self
        for child in self.children:
            for n in child.iternodes(f):
                yield n

    def iterleaves(self):
        return self.iternodes(lambda x:x.isleaf)

    def preiter(self, f=None):
        for n in self.iternodes(f=f):
            yield n

    def postiter(self, f=None):
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
        v = self.find(f, *args, **kwargs)
        try:
            return v.next()
        except StopIteration:
            return None

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

    def excise(self):
        "for 'knees': remove self from between parent and single child"
        assert self.parent
        assert len(self.children)==1
        p = self.parent
        c = self.children[0]
        if c.length is not None and self.length is not None:
            c.length += self.length
        c.prune()
        self.prune()
        p.add_child(c)
        return p

    def graft(self, node):
        parent = self.parent
        parent.remove_child(self)
        n = Node()
        n.add_child(self)
        n.add_child(node)
        parent.add_child(n)

    ## def leaf_distances(self, store=None, measure="length"):
    ##     """
    ##     for each internal node, calculate the distance to each leaf,
    ##     measured in branch length or internodes
    ##     """
    ##     if store is None:
    ##         store = {}
    ##     leaf2len = {}
    ##     if self.children:
    ##         for child in self.children:
    ##             if measure == "length":
    ##                 dist = child.length
    ##             elif measure == "nodes":
    ##                 dist = 1
    ##             child.leaf_distances(store, measure)
    ##             if child.isleaf:
    ##                 leaf2len[child] = dist
    ##             else:
    ##                 for k, v in store[child].items():
    ##                     leaf2len[k] = v + dist
    ##     else:
    ##         leaf2len[self] = {self: 0}
    ##     store[self] = leaf2len
    ##     return store

    def leaf_distances(self, measure="length"):
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
        """
        n = self.parent
        while 1:
            if n is None: raise StopIteration
            yield n
            if n.isroot or (end and n == end) or (stop and stop(n)):
                raise StopIteration
            n = n.parent

    def rootpath_length(self, end=None):
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

    def reroot_orig(self, newroot):
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

    def reroot(self, newroot):
        newroot = self[newroot]
        assert newroot in self
        self.isroot = False
        n = newroot
        v = list(n.rootpath())
        v.reverse()
        for node in (v+[n])[1:]:
            # node is current node; cp is current parent
            cp = node.parent
            cp.remove_child(node)
            node.add_child(cp)
            cp.length = node.length
            cp.label = node.label
        newroot.isroot = True
        return newroot

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


reroot = Node.reroot

def index(node, n=0, d=0):
    """
    recursively attach 'next', 'back', (and 'left', 'right'), 'ni',
    'ii', 'pi', and 'node_depth' attributes to nodes
    """
    node.next = node.left = n
    if not node.parent:
        node.node_depth = d
    else:
        node.node_depth = node.parent.node_depth + 1
    n += 1
    for i, c in enumerate(node.children):
        if i > 0:
            n = node.children[i-1].back + 1
        index(c, n)

    if node.children:
        node.back = node.right = node.children[-1].back + 1
    else:
        node.back = node.right = n
    return node.back

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
            results[node] = sum(results[child] for child in node.children)
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

def write(node, outfile=None, format="newick", length_fmt=":%g",
          clobber=False):
    if format=="newick" or ((type(outfile) in types.StringTypes) and
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
        if type(outfile) in types.StringTypes:
            if not clobber:
                assert not os.path.isfile(outfile), "File '%s' exists! (Set clobber=True to overwrite)" % outfile
            flag = True
            outfile = file(outfile, "w")
        outfile.write(s)
        if flag:
            outfile.close()
    return s
    
def read(data, format=None, treename=None, ttable=None):
    """
    Read a single tree from *data*, which can be a Newick string, a
    file name, or a file-like object with `tell` and 'read`
    methods. *treename* is an optional string that will be attached to
    all created nodes.

    Returns: *root*, the root node.
    """
    import newick
    StringTypes = types.StringTypes
    
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
                return newick.parse(file(data), treename=treename,
                                    ttable=ttable)
            else:
                return newick.parse(data, ttable=ttable)

        elif (hasattr(data, "tell") and hasattr(data, "read")):
            treename = strip(getattr(data, "name", None))
            return newick.parse(data, treename=treename, ttable=ttable)
    elif format == "nexus-dendropy":
        import dendropy
        if type(data) in StringTypes:
            if os.path.isfile(data):
                treename = strip(data)
                return newick.parse(
                    str(dendropy.Tree.get_from_path(data, "nexus")),
                    treename=treename
                    )
            else:
                return newick.parse(
                    str(dendropy.Tree.get_from_string(data, "nexus"))
                    )

        elif (hasattr(data, "tell") and hasattr(data, "read")):
            treename = strip(getattr(data, "name", None))
            return newick.parse(
                str(dendropy.Tree.get_from_stream(data, "nexus")),
                treename=treename
                )
        else:
            pass

    elif format == "nexus":
        if type(data) in StringTypes:
            if os.path.isfile(data):
                with open(data) as infile:
                    rec = newick.nexus_iter(infile).next()
                    if rec: return rec.parse()
            else:
                rec = newick.nexus_iter(StringIO(data)).next()
                if rec: return rec.parse()
        else:
            rec = newick.nexus_iter(data).next()
            if rec: return rec.parse()
    else:
        # implement other tree formats here (nexus, nexml etc.)
        raise IOError, "format '%s' not implemented yet" % format

    raise IOError, "unable to read tree from '%s'" % data

def readmany(data, format="newick"):
    "Iterate over trees from a source."
    if type(data) in types.StringTypes:
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
        raise Exception, "format '%s' not recognized" % format
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
    
        
