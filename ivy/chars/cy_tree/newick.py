"""
Parse newick strings.

The function of interest is `parse`, which returns the root node of
the parsed tree.
"""
import string, sys, re, shlex, types, itertools
import numpy
import nexus
from cStringIO import StringIO
from pprint import pprint

## def read(s):
##     try:
##         s = file(s).read()
##     except:
##         try:
##             s = s.read()
##         except:
##             pass
##     return parse(s)

LABELCHARS = '-.|/?#&'
META = re.compile(r'([^,=\s]+)\s*=\s*(\{[^=}]*\}|"[^"]*"|[^,]+)?')

def add_label_chars(chars):
    global LABELCHARS
    LABELCHARS += chars

class Tokenizer(shlex.shlex):
    """Provides tokens for parsing newick strings."""
    def __init__(self, infile):
        global LABELCHARS
        shlex.shlex.__init__(self, infile, posix=False)
        self.commenters = ''
        self.wordchars = self.wordchars+LABELCHARS
        self.quotes = "'"

    def parse_embedded_comment(self):
        ws = self.whitespace
        self.whitespace = ""
        v = []
        while 1:
            token = self.get_token()
            if token == '':
                sys.stdout.write('EOF encountered mid-comment!\n')
                break
            elif token == ']':
                break
            elif token == '[':
                self.parse_embedded_comment()
            else:
                v.append(token)
        self.whitespace = ws
        return "".join(v)
        ## print "comment:", v

def parse(data, ttable=None, treename=None):
    """
    Parse a newick string.

    *data* is any file-like object that can be coerced into shlex, or
    a string (converted to StringIO)

    *ttable* is a dictionary mapping node labels in the newick string
     to other values.

    Returns: the root node.
    """
    from cy_tree import Node, Tree

    if type(data) in types.StringTypes:
        data = StringIO(data)

    start_pos = data.tell()
    tokens = Tokenizer(data)

    node = None; root = None
    lp=0; rp=0; rooted=1

    previous = None

    ni = 0 # node id counter (preorder) - zero-based indexing
    li = 0 # leaf index counter
    ii = 0 # internal node index counter
    pi = 0 # postorder sequence
    while 1:
        token = tokens.get_token()
        if token == ';' or token == tokens.eof:
            assert lp == rp, \
                   "unbalanced parentheses in tree description: (%s, %s)" \
                   % (lp, rp)
            break

        # internal node
        elif token == '(':
            lp = lp+1
            newnode = Node()
            newnode.ni = ni; ni += 1
            ## newnode.isleaf = False
            newnode.treename = treename
            if node:
                if node.children: newnode.left = node.children[-1].right+1
                else: newnode.left = node.left+1
                node.add_child(newnode)
            else:
                newnode.left = 1; newnode.right = 2
            newnode.right = newnode.left+1
            node = newnode

        elif token == ')':
            rp = rp+1
            node = node.parent
            if node.children:
                node.right = node.children[-1].right + 1

        elif token == ',':
            node = node.parent
            if node.children:
                node.right = node.children[-1].right + 1

        # branch length
        elif token == ':':
            token = tokens.get_token()
            if token == '[':
                node.length_comment = tokens.parse_embedded_comment()
                token = tokens.get_token()

            if not (token == ''):
                try: brlen = float(token)
                except ValueError:
                    raise ValueError, ("invalid literal for branch length, "
                                       "'%s'" % token)
            else:
                raise 'NewickError', \
                      'unexpected end-of-file (expecting branch length)'

            node.length = brlen
        # comment
        elif token == '[':
            node.comment = tokens.parse_embedded_comment()
            if node.comment[0] == '&':
                # metadata
                meta = META.findall(node.comment[1:])
                if meta:
                    node.meta = {}
                    for k, v in meta:
                        v = eval(v.replace('{','(').replace('}',')'))
                        node.meta[k] = v

        # leaf node or internal node label
        else:
            if previous != ')': # leaf node
                if ttable:
                    try:
                        ttoken = (ttable.get(int(token)) or
                                  ttable.get(token))
                    except ValueError:
                        ttoken = ttable.get(token)
                    if ttoken:
                        token = ttoken
                newnode = Node()
                newnode.ni = ni; ni += 1
                newnode.label = "_".join(token.split()).replace("'", "")
                ## newnode.isleaf = True
                if node.children: newnode.left = node.children[-1].right+1
                else: newnode.left = node.left+1
                newnode.right = newnode.left+1
                newnode.treename = treename
                node.add_child(newnode)
                node = newnode
            else: # label
                if ttable:
                    node.label = ttable.get(token, token)
                else:
                    node.label = token

        previous = token
    ## node.isroot = True
    return node

def parse_ampersand_comment(s):
    import pyparsing
    pyparsing.ParserElement.enablePackrat()
    from pyparsing import Word, Literal, QuotedString, CaselessKeyword, \
         OneOrMore, Group, Optional, Suppress, Regex, Dict
    word = Word(string.letters+string.digits+"%_")
    key = word.setResultsName("key") + Suppress("=")
    single_value = (Word(string.letters+string.digits+"-.") |
                    QuotedString("'") |
                    QuotedString('"'))
    range_value = Group(Suppress("{") +
                        single_value.setResultsName("min") +
                        Suppress(",") +
                        single_value.setResultsName("max") +
                        Suppress("}"))
    pair = (key + (single_value | range_value).setResultsName("value"))
    g = OneOrMore(pair)
    d = []
    for x in g.searchString(s):
        v = x.value
        if type(v) == str:
            try: v = float(v)
            except ValueError: pass
        else:
            try: v = map(float, v.asList())
            except ValueError: pass
        d.append((x.key, v))
    return d

def nexus_iter(infile):
    import pyparsing
    pyparsing.ParserElement.enablePackrat()
    from pyparsing import Word, Literal, QuotedString, CaselessKeyword, \
         OneOrMore, Group, Optional, Suppress, Regex, Dict
    ## beginblock = Suppress(CaselessKeyword("begin") +
    ##                       CaselessKeyword("trees") + ";")
    ## endblock = Suppress((CaselessKeyword("end") |
    ##                      CaselessKeyword("endblock")) + ";")
    comment = Optional(Suppress("[&") + Regex(r'[^]]+') + Suppress("]"))
    ## translate = CaselessKeyword("translate").suppress()
    name = Word(string.letters+string.digits+"_.") | QuotedString("'")
    ## ttrec = Group(Word(string.digits).setResultsName("number") +
    ##               name.setResultsName("name") +
    ##               Optional(",").suppress())
    ## ttable = Group(translate + OneOrMore(ttrec) + Suppress(";"))
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

    def not_begin(s): return s.strip().lower() != "begin trees;"
    def not_end(s): return s.strip().lower() not in ("end;", "endblock;")
    def parse_ttable(f):
        ttable = {}
        while True:
            s = f.next().strip()
            if not s: continue
            if s.lower() == ";": break
            if s[-1] == ",": s = s[:-1]
            k, v = s.split()
            ttable[k] = v
            if s[-1] == ";": break
        return ttable

    # read lines between "begin trees;" and "end;"
    f = itertools.takewhile(not_end, itertools.dropwhile(not_begin, infile))
    s = f.next().strip().lower()
    if s != "begin trees;":
        print sys.stderr, "Expecting 'begin trees;', got %s" % s
        raise StopIteration
    ttable = {}
    while True:
        try: s = f.next().strip()
        except StopIteration: break
        if not s: continue
        if s.lower() == "translate":
            ttable = parse_ttable(f)
            print "ttable: %s" % len(ttable)
        elif s.split()[0].lower()=='tree':
            match = tree.parseString(s)
            yield nexus.Newick(match, ttable)

## def test():
##     with open("/home/rree/Dropbox/pedic-comm-amnat/phylo/beast-results/"
##               "simple_stigma.trees.log") as f:
##         for rec in nexus_iter(f):
##             r = parse(rec.newick, ttable=rec.ttable)
##             for x in r: print x, x.comments

def test_parse_comment():
    v = (("height_median=1.1368683772161603E-13,height=9.188229043880098E-14,"
          "height_95%_HPD={5.6843418860808015E-14,1.7053025658242404E-13},"
          "height_range={0.0,2.8421709430404007E-13}"),
         "R", "lnP=-154.27154502342688,lnP=-24657.14341301901",
         'states="T-lateral"')
    for s in v:
        print "input:", s
        print dict(parse_ampersand_comment(s))
