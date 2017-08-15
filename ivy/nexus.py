from __future__ import print_function
import itertools
from . import newick

class Newick(object):
    """
    convenience class for storing the results of a newick tree
    record from a nexus file, as parsed by newick.nexus_iter
    """
    def __init__(self, parse_results=None, ttable={}):
        self.name = ""
        self.comment = ""
        self.root_comment = ""
        self.newick = ""
        self.ttable = ttable
        if parse_results: self.populate(parse_results)

    def populate(self, parse_results, ttable={}):
        self.name = parse_results.tree_name
        self.comment = parse_results.tree_comment
        self.root_comment = parse_results.root_comment
        self.newick = parse_results.newick
        if ttable: self.ttable = ttable

    def parse(self, newick=newick):
        assert self.newick
        self.root = newick.parse(
            self.newick, ttable=self.ttable, treename=self.name
            )
        return self.root

def fetchaln(fname):
    """Fetch alignment"""
    from Bio.Nexus import Nexus
    n = Nexus.Nexus(fname)
    return n

def split_blocks(infile):
    try:
        from cStringIO import StringIO
    except:
        from io import StringIO
    dropwhile = itertools.dropwhile; takewhile = itertools.takewhile
    blocks = []
    not_begin = lambda s: not s.lower().startswith("begin")
    not_end = lambda s: not s.strip().lower() in ("end;", "endblock;")
    while 1:
        f = takewhile(not_end, dropwhile(not_begin, infile))
        try:
            b = f.next().split()[-1][:-1]
            blocks.append((b, StringIO("".join(list(f)))))
        except StopIteration:
            break
    return blocks

def parse_treesblock(infile):
    import string
    from pyparsing import Optional, Word, Regex, CaselessKeyword, Suppress
    from pyparsing import QuotedString
    comment = Optional(Suppress("[&") + Regex(r'[^]]+') + Suppress("]"))
    name = Word(alphanums+"_") | QuotedString("'")
    newick = Regex(r'[^;]+;')
    tree = (CaselessKeyword("tree").suppress() +
            Optional("*").suppress() +
            name.setResultsName("tree_name") +
            comment.setResultsName("tree_comment") +
            Suppress("=") +
            comment.setResultsName("root_comment") +
            newick.setResultsName("newick"))
    ## treesblock = Group(beginblock +
    ##                    Optional(ttable.setResultsName("ttable")) +
    ##                    Group(OneOrMore(tree)) +
    ##                    endblock)

    def parse_ttable(f):
        ttable = {}
        while True:
            s = f.next().strip()
            if s.lower() == ";":
                break
            if s[-1] in ",;":
                s = s[:-1]
            k, v = s.split()
            ttable[k] = v
            if s[-1] == ";":
                break
        return ttable

    ttable = {}
    while True:
        try:
            s = infile.next().strip()
        except StopIteration:
            break
        if s.lower() == "translate":
            ttable = parse_ttable(infile)
            # print("ttable: %s" % len(ttable))
        else:
            match = tree.parseString(s)
            yield Newick(match, ttable)

def iter_trees(infile):
    import pyparsing
    pyparsing.ParserElement.enablePackrat()
    from pyparsing import (
        Word, Literal, QuotedString, CaselessKeyword, CharsNotIn,
        OneOrMore, Group, Optional, Suppress, Regex, Dict, ZeroOrMore,
        alphanums, nums)
    comment = Optional(Suppress("[&") + Regex(r'[^]]+') + Suppress("]"))
    name = Word(alphanums+"_.") | QuotedString("'")
    newick = Regex(r'[^;]+;')
    tree = (CaselessKeyword("tree").suppress() +
            Optional("*").suppress() +
            name.setResultsName("tree_name") +
            comment.setResultsName("tree_comment") +
            Suppress("=") +
            comment.setResultsName("root_comment") +
            newick.setResultsName("newick"))

    def not_begin(s):
        # print('not_begin', s)
        return s.strip().lower() != "begin trees;"
    def not_end(s):
        # print('not_end', s)
        return s.strip().lower() not in ("end;", "endblock;")
    def parse_ttable(f):
        ttable = {}
        # com = Suppress('[') + ZeroOrMore(CharsNotIn(']')) + Suppress(']')
        com = Suppress('[' + ZeroOrMore(CharsNotIn(']') + ']'))
        while True:
            s = next(f).strip()
            if not s:
                continue
            s = com.transformString(s).strip()
            if s.lower() == ";":
                break
            b = False
            if s[-1] in ",;":
                if s[-1] == ';':
                    b = True
                s = s[:-1]
            # print(s)
            k, v = s.split()
            ttable[k] = v
            if b:
                break
        return ttable

    # read lines between "begin trees;" and "end;"
    f = itertools.takewhile(not_end, itertools.dropwhile(not_begin, infile))
    s = next(f).strip().lower()
    if s != "begin trees;":
        print("Expecting 'begin trees;', got %s" % s, file=sys.stderr)
        raise StopIteration
    ttable = {}
    while True:
        try:
            s = next(f).strip()
        except StopIteration:
            break
        if not s:
            continue
        if s.lower() == "translate":
            ttable = parse_ttable(f)
            # print "ttable: %s" % len(ttable)
        elif s.split()[0].lower()=='tree':
            match = tree.parseString(s)
            yield Newick(match, ttable)
