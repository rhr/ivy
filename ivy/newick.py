"""
Parse newick strings.

The function of interest is `parse`, which returns the root node of
the parsed tree.
"""
import string, sys
from shlex import shlex
import types
from cStringIO import StringIO

## def read(s):
##     try:
##         s = file(s).read()
##     except:
##         try:
##             s = s.read()
##         except:
##             pass
##     return parse(s)

class Tokenizer(shlex):
    """Provides tokens for parsing newick strings."""
    def __init__(self, infile):
        shlex.__init__(self, infile)
        self.commenters = ''
        self.wordchars = self.wordchars+"-.|"
        self.quotes = "'"

    def parse_comment(self):
        while 1:
            token = self.get_token()
            if token == '':
                sys.stdout.write('EOF encountered mid-comment!\n')
                break
            elif token == ']':
                break
            elif token == '[':
                self.parse_comment()
            else:
                pass

def parse(data, ttable=None, treename=None):
    """
    Parse a newick string.

    *data* is any file-like object that can be coerced into shlex, or
    a string (converted to StringIO)

    *ttable* is a dictionary mapping node labels in the newick string
     to other values.

    Returns: the root node.
    """
    from tree import Node
    
    if type(data) in types.StringTypes:
        data = StringIO(data)
    
    start_pos = data.tell()
    tokens = Tokenizer(data)

    node = None; root = None
    lp=0; rp=0; rooted=1

    previous = None

    i = 1 # node id counter
    while 1:
        token = tokens.get_token()
        #print token,
        if token == ';' or token == '':
            assert lp == rp, \
                   "unbalanced parentheses in tree description: (%s, %s)" \
                   % (lp, rp)
            break

        # internal node
        elif token == '(':
            lp = lp+1
            newnode = Node()
            newnode.id = i; i += 1
            newnode.isleaf = False
            newnode.treename = treename
            if node:
                node.add_child(newnode)
            node = newnode

        elif token == ')':
            rp = rp+1
            node = node.parent
            
        elif token == ',':
            node = node.parent
            
        # branch length
        elif token == ':':
            token = tokens.get_token()
            if token == '[':
                tokens.parse_comment()
                token = tokens.get_token()

            if not (token == ''):
                try:
                    brlen = float(token)
                except ValueError:
                    raise ValueError, "invalid literal for branch length, '%s'" % token
            else:
                raise 'NewickError', \
                      'unexpected end-of-file (expecting branch length)'

            node.length = brlen
        # comment
        elif token == '[':
            tokens.parse_comment()

        # leaf node or internal node label
        else:
            if previous != ')': # leaf node
                if ttable:
                    try:
                        ttoken = ttable.get(int(token))
                    except ValueError:
                        ttoken = ttable.get(token)
                    if ttoken:
                        token = ttoken
                newnode = Node()
                newnode.id = i; i += 1
                newnode.label = "_".join(token.split()).replace("'", "")
                newnode.isleaf = True
                newnode.treename = treename
                node.add_child(newnode)
                node = newnode
            else: # label
                if ttable:
                    node.label = ttable.get(token, token)
                else:
                    node.label = token

        previous = token
    node.isroot = True
    return node

def string(node, length_fmt=":%s", end=True, newline=True):
    "Recursively create a newick string from node."
    if not node.isleaf:
        node_str = "(%s)%s" % \
                   (",".join([ string(child, length_fmt, False, newline) \
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
        if not newline:
            semicolon = ";"
        else:
            semicolon = ";\n"
    s = "%s%s%s" % (node_str, length_str, semicolon)
    return s
