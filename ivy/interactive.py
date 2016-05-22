#!/usr/bin/env ipython
# -*- coding: utf-8 -*-
"""
Adds to the interactive IPython/pylab environment
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys, os, re
import ivy
import ivy.vis

def readtree(data, *args, **kwargs): return ivy.tree.read(data, *args, **kwargs)

def readaln(data, *args, **kwargs): return ivy.align.read(data, *args, **kwargs)

def treefig(*args, **kwargs):
    from ivy.vis.treevis import TreeFigure
    if len(args) == 1:
        fig = TreeFigure(args[0], **kwargs)
    else:
        fig = MultiTreeFigure(**kwargs)
        for arg in args:
            print(arg)
            fig.add(arg)
    fig.show()
    return fig

def alnfig(*args, **kwargs):
    from ivy.vis import AlignmentFigure
    return AlignmentFigure(*args, **kwargs)

def __maketree(self, s):
    import os#, IPython
    words = s.split()
    treename = "root"
    fname = None
    if words:
        treename = words.pop(0)
        if words and os.path.isfile(words[0]):
            fname = words.pop(0)

    if not fname:
        ## msg = "\n".join([
        ##     "Name of tree file",
        ##     "(Try dragging one into the terminal):\n"
        ##     ])
        msg = "Enter the name of a tree file or a newick string:\n"
        fname = input(msg).strip()

    quotes = ["'", '"']
    if fname and fname[0] in quotes:
        fname = fname[1:]
    if fname and fname[-1] in quotes:
        fname = fname[:-1]
    if fname:
        try:
            ## root = ivy.tree.read(fname)
            ## IPython.ipapi.get().to_user_ns({treename:root})
            cmd = "%s = ivy.tree.read('%s')" % (treename, fname)
            get_ipython().ex(cmd)
            print("Tree parsed and assigned to variable '%s'" % treename)
        except:
            print("Unable to parse tree file '%s'" % fname)
    else:
        print("Cancelled")

def __node_completer(self, event):
    symbol = event.symbol
    s = event.line
    if symbol:
        s = s[:-len(symbol)]
    quote = ""
    if s and s[-1] in ["'", '"']:
        quote = s[-1]
        s = s[:-1]
    #base = (re.findall(r'(\w+)\[\Z', s) or [None])[-1]
    base = "".join((re.findall(r'(\w+\.\w*)?(\.)?(\w+)\[\Z', s) or [""])[-1])
    ## print "symbol:", symbol
    ## print "line:", event.line
    ## print "s:", s
    ## print "quote:", quote
    ## print "base:", base
    ## print "obj:", self._ofind(base).get("obj")

    obj = None
    if base:
        obj = self._ofind(base).get("obj")
    ## print '\n'
    ## print 'base', base
    ## print 'obj', obj
    if obj and isinstance(obj, ivy.tree.Node):
        completions = ["'"]
        if quote:
            completions = sorted([ x.label for x in obj.labeled() ])
        return completions

    raise IPython.core.error.TryNext()

try:
    ## import IPython
    IP = get_ipython() #IPython.ipapi.get()
    if IP:
        #IP.expose_magic("maketree", __maketree)
        IP.define_magic("maketree", __maketree)
        ## IP.set_hook(
        ##     "complete_command", __node_completer, re_key=r'\[*'
        ##     )
        IP.set_hook(
            "complete_command", __node_completer,
            re_key='.+[[]([\']|["])*\w*$'
            )

except:
    print(sys.exc_info()[0])
    sys.stderr.write("Magic commands and completers requires IPython >= 0.11\n")

## if __name__ == "__main__":
##     if len(sys.argv) > 1:
##         for fname in sys.argv[1:]:
##             if os.path.isfile(fname):
##                 execfile(fname)
