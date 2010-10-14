#!/usr/bin/env python
"""
Use the ``ivy`` module to graft the angiosperm megatree from
<http://svn.phylodiversity.net>.  The config file and source trees are
fetched directly from the server.
"""
import sys, os
from urllib2 import urlopen
import ivy

flush = sys.stdout.flush
URLBASE = "http://svn.phylodiversity.net/tot/trees"
config = urlopen(URLBASE+"/makemega.config")
name2clade = {}
root = None

for line in [ x.strip() for x in config
              if x.strip() and (not x.startswith("#")) ]:
    isroot = False
    if line.startswith("ROOT"):
        line = line.split("=")[-1].strip()
        isroot = True
    url = "%s/%s.new" % (URLBASE, line)
    print "Fetch:", url; flush()
    clade = ivy.tree.read(urlopen(url).read())
    if isroot:
        root = clade
    else:
        clade.isroot = False
        name = clade.label
        name2clade[name] = clade
        
# at time of writing, makemega.config lacked an angiosperm entry
if "angiosperms" not in name2clade:
    url = URLBASE+"/angiosperms_apg2009.new"
    clade = ivy.tree.read(urlopen(url).read())
    clade.isroot = False
    name2clade["angiosperms"] = clade

assert root, "ROOT not in config!"

# graft the tree
while name2clade:
    found = False
    for leaf in root.leaves():
        branch = name2clade.pop(leaf.label, None)
        if branch:
            print "Graft:", leaf.label; flush()
            found = True
            leaf.isleaf = False
            parent = leaf.prune()
            parent.add_child(branch)
    if not found:
        break

# capitalize names and tidy up
for node in root.labeled():
    node.label = node.label.capitalize()
root.ladderize()

# show the tree
from ivy.interactive import *
fig = treefig(root, branchlabels=True)
