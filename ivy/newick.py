"""
Parse newick strings.

The function of interest is `parse`, which returns the root node of
the parsed tree.
"""
# from __future__ import print_function, absolute_import, division, unicode_literals
import string, sys, re, shlex, logging, pdb
try:
    from cStringIO import StringIO
except:
    from io import StringIO

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

class Error(Exception):
    pass

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
            logging.debug('[embed] token = {}'.format(token))
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

    Args:
        data: Any file-like object that can be coerced into shlex, or
          a string (converted to StringIO)
        ttable (dict): Mapping of node labels in the newick string
          to other values.

    Returns:
        Node: The root node.
    """
    from .tree import Node

    if isinstance(data, str):
        data = StringIO(data)

    start_pos = data.tell()
    tokens = Tokenizer(data)

    node = None; root = None
    lp=0; rp=0; rooted=1

    previous = None

    ni = 0  # node id counter (preorder) - zero-based indexing
    li = 0  # leaf index counter
    ii = 0  # internal node index counter
    pi = 0  # postorder sequence
    while 1:
        token = tokens.get_token()
        logging.debug('node = {}, token = {}'.format(node, token))
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
            newnode.isleaf = False
            newnode.ii = ii; ii += 1
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
            if previous == '(':
                # edge case - unlabeled tip
                newnode = Node()
                newnode.ni = ni; ni += 1
                newnode.pi = pi; pi += 1
                newnode.label = ''
                newnode.isleaf = True
                newnode.li = li; li += 1
                if node.children:
                    newnode.left = node.children[-1].right+1
                else:
                    newnode.left = node.left+1
                newnode.right = newnode.left+1
                newnode.treename = treename
                node.add_child(newnode)
                node = newnode
            rp = rp+1
            node = node.parent
            node.pi = pi; pi += 1
            if node.children:
                node.right = node.children[-1].right + 1

        elif token == ',':
            _n = node
            node = node.parent
            if node.children:
                node.right = node.children[-1].right + 1

        # branch length
        elif token == ':':
            token = tokens.get_token()
            logging.debug('node = {}, token = {}'.format(node, token))
            if token == '[':
                node.length_comment = tokens.parse_embedded_comment()
                token = tokens.get_token()
                logging.debug('node = {}, token = {}'.format(node, token))

            if not (token == ''):
                try:
                    brlen = float(token)
                except ValueError as exc:
                    brlen = None
                    if token == '{':  # simmap history
                        v = []
                        while 1:
                            state = tokens.get_token()
                            comma = tokens.get_token()
                            assert comma == ',', comma
                            seg = float(tokens.get_token())
                            token = tokens.get_token()
                            v.append((state, seg))
                            if token == '}':
                                break
                        node.simmap = v
                    else:
                        raise ValueError(
                            "invalid literal for branch length, '{}'".format(token))
            else:
                raise Error('unexpected end-of-file (expecting branch length)')

            node.length = brlen
        # comment
        elif token == '[':
            node.comment = tokens.parse_embedded_comment()
            if node.comment[0] == '&':
                # metadata
                meta = META.findall(node.comment[1:])
                if meta:
                    for k, v in meta:
                        try:
                            v = eval(v.replace('{','(').replace('}',')'))
                        except NameError:
                            v = str(v)
                        node.meta[k] = v

        # leaf node or internal node label
        else:
            if previous != ')':  # leaf node
                if ttable:
                    try:
                        ttoken = (ttable.get(int(token)) or ttable.get(token))
                    except ValueError:
                        ttoken = ttable.get(token)
                    if ttoken:
                        token = ttoken
                newnode = Node()
                newnode.ni = ni; ni += 1
                newnode.pi = pi; pi += 1
                newnode.label = "_".join(token.split()).replace("'", "")
                newnode.isleaf = True
                newnode.li = li; li += 1
                if node.children:
                    newnode.left = node.children[-1].right+1
                else:
                    newnode.left = node.left+1
                newnode.right = newnode.left+1
                newnode.treename = treename
                node.add_child(newnode)
                node = newnode
            else:  # label
                if ttable:
                    node.label = ttable.get(token, token)
                else:
                    node.label = token

        previous = token
    node.isroot = True
    return node

## def string(node, length_fmt=":%s", end=True, newline=True):
##     "Recursively create a newick string from node."
##     if not node.isleaf:
##         node_str = "(%s)%s" % \
##                    (",".join([ string(child, length_fmt, False, newline) \
##                                for child in node.children ]),
##                     node.label or ""
##                     )
##     else:
##         node_str = "%s" % node.label

##     if node.length is not None:
##         length_str = length_fmt % node.length
##     else:
##         length_str = ""

##     semicolon = ""
##     if end:
##         if not newline:
##             semicolon = ";"
##         else:
##             semicolon = ";\n"
##     s = "%s%s%s" % (node_str, length_str, semicolon)
##     return s

## def from_nexus(infile, bufsize=None):
##     bufsize = bufsize or 1024*5000
##     TTABLE = re.compile(r'\btranslate\s+([^;]+);', re.I | re.M)
##     TREE = re.compile(r'\btree\s+([_.\w]+)\s*=[^(]+(\([^;]+;)', re.I | re.M)
##     s = infile.read(bufsize)
##     ttable = TTABLE.findall(s) or None
##     if ttable:
##         items = [ shlex.split(line) for line in ttable[0].split(",") ]
##         ttable = dict([ (k, v.replace(" ", "_")) for k, v in items ])
##     trees = TREE.findall(s)
##     ## for i, t in enumerate(trees):
##     ##     t = list(t)
##     ##     if ttable:
##     ##         t[1] = "".join(
##     ##             [ ttable.get(x, "_".join(x.split()).replace("'", ""))
##     ##               for x in shlex.shlex(t[1]) ]
##     ##             )
##     ##     trees[i] = t
##     ## return trees
##     return ttable, trees

def parse_ampersand_comment(s):
    import pyparsing
    pyparsing.ParserElement.enablePackrat()
    from pyparsing import Word, Literal, QuotedString, CaselessKeyword, \
         OneOrMore, Group, Optional, Suppress, Regex, Dict
    word = Word(string.ascii_letters+string.digits+"%_")
    key = word.setResultsName("key") + Suppress("=")
    single_value = (Word(string.ascii_letters+string.digits+"-.") |
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
        if isinstance(v, str):
            try: v = float(v)
            except ValueError: pass
        else:
            try: v = map(float, v.asList())
            except ValueError: pass
        d.append((x.key, v))
    return d

# def test_parse_comment():
#     v = (("height_median=1.1368683772161603E-13,height=9.188229043880098E-14,"
#           "height_95%_HPD={5.6843418860808015E-14,1.7053025658242404E-13},"
#           "height_range={0.0,2.8421709430404007E-13}"),
#          "R", "lnP=-154.27154502342688,lnP=-24657.14341301901",
#          'states="T-lateral"')
#     for s in v:
#         print "input:", s
#         print dict(parse_ampersand_comment(s))
