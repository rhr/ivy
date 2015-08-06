.. _users-guide

************
User's Guide
************

``ivy`` is designed to be used both as a library and interactively, in
the IPython shell.  `IPython <http://ipython.scipy.org>`_ is an
enhanced Python interpreter designed for interactive and exploratory
scientific computing.

This `screencast <http://vimeo.com/23646898>`_ demonstrates some basic
concepts:

.. raw:: html

    <iframe
     src="http://player.vimeo.com/video/23646898?title=0&amp;byline=0&amp;portrait=0"
     width="400" height="208" frameborder="0"></iframe>


    <p><a href="http://vimeo.com/23646898" target="_blank">Watch in
     full resolution (opens in new window)</a></p>

Quickstart
==========

Starting the shell
------------------

The ``-pylab`` option starts IPython in a mode that imports a number
of functions from matplotlib and numpy, and allows interactive
plotting.

.. code-block:: bash

    $ ipython -pylab

In interactive mode, ``ivy`` provides some useful shortcut functions
(e.g., readtree, readaln, treefig, alnfig) that you will typically
want to import as follows.

.. sourcecode:: ipython

    In [1]: from ivy.interactive import *

Viewing a tree
--------------

Assuming you have started the shell,

.. sourcecode:: ipython

   In [2]: s = "examples/primates.newick"
   In [3]: fig = treefig(s)

Here, the variable *s* can be a newick string, the name (path) of a
file containing a newick tree, or an open file containing a newick
string.  Note that file paths are completed dynamically in ipython by
hitting the <TAB> key, making it easy to find files with little
typing.
   
A new window should appear, controlled by the variable *fig*.  View
the help for *fig*::

   fig?

Trees in Ivy
=============

Ivy does not have a tree class per se; rather trees in Ivy exist as collections
of nodes. in Ivy, a Node is a class that contains information about that node. 
Nodes are rooted and recursively contain their children. Functions
in Ivy act directly on Node objects. Nodes support Python idioms such as `in`, 
`[`, iteration, etc. This guide will cover how to read, view, navigate, modify, 
and write trees in Ivy.

Reading Trees
=============

You can read in trees using Ivy's `tree.read` function. This function supports 
newick and nexus files. The tree.read function can take a file name, a file
object, or a Newick string as input. The output of this function is the root
node of the tree.

.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: f = open("examples/primates.newick")
    In [*]: r = ivy.tree.read(f)
    In [*]: f.close()
    In [*]: r = ivy.tree.read("examples/primates.newick")
    In [*]: r = ivy.tree.read(
                "((((Homo:0.21,Pongo:0.21)A:0.28,Macaca:0.49)"
                "B:0.13,Ateles:0.62)C:0.38,Galago:1.00)root;")
    In [*]: # These three methods are identical

You can copy a read tree using the `copy` method on the root node. Node
objects are mutable, so this method is preferred over r2 = r if you want
to create a deep copy.

.. sourcecode:: ipython

    In [*]: r2 = r.copy(recurse=True) # If recurse=False, won't copy children etc.

.. warning::

    As of now, the copy function does not produce a complete tree: the nodes are not
    properly connected to each other 

.. sourcecode:: ipython

    In [*]: print r2["A"].parent
    None

Viewing Trees
=============

There are a number of ways you can view trees in Ivy. For a simple display
without needing to create a plot, Ivy can create ascii trees that can be
printed to the console.

.. sourcecode:: ipython

    In [*]: print r.ascii # You can use the ascii method on root nodes.
                                   ---------+ Homo   
                          --------A+                 
                 --------B+        ---------+ Pongo  
                 :        :                          
        --------C+        ------------------+ Macaca 
        :        :                                   
    root+        ---------------------------+ Ateles 
        :                                            
        ------------------------------------+ Galago 
 
For a more detailed and interactive tree, Ivy can create a plot using 
`Matplotlib`. More detail about visualization using Matplotlib will follow
later in the guide.

.. sourcecode:: ipython

    In [*]: import ivy.vis
    In [*]: fig = ivy.vis.tree.TreeFigure(r)
    In [*]: fig.show()

You can also create a plot using`Bokeh`.

.. sourcecode:: ipython

    In [*]: import ivy.vis.bokehtree
    In [*]: fig2 = ivy.vis.bokehtree.BokehTree(r)
    In [*]: fig2.drawtree()

.. TODO: embed plot images

Navigating Trees
================

A node in Ivy is a container. It recursively contains its descendants, 
as well as itself. You can navigate a tree using the Python idioms that
you are used to.

Let's start by iterating over all of the children contained within the root 
node. By default, iteration over a node happens in preorder sequence, starting
with the root node.

.. sourcecode:: ipython

    In [*]: len(r)
    Out[*]: 9 # Length of a node = number of descendants + self
    In [*]: for node in r:
                print node # Default is preorder sequence
    Node(139624003155728, root, 'root')
    Node(139624003155536, 'C')
    Node(139624003155600, 'B')
    Node(139624003155664, 'A')
    Node(139624003155792, leaf, 'Homo')
    Node(139624003155856, leaf, 'Pongo')
    Node(139624003155920, leaf, 'Macaca')
    Node(139624003155984, leaf, 'Ateles')
    Node(139624003156048, leaf, 'Galago')
    In [*]: for node in r.preiter: 
                print node # Same as above
    Node(140144824314320, root, 'root')
    Node(140144824314384, 'C')
    Node(140144824314448, 'B')
    Node(140144824314512, 'A')
    Node(140144824314576, leaf, 'Homo')
    Node(140144824314192, leaf, 'Pongo')
    Node(140144824314256, leaf, 'Macaca')
    Node(140144824314640, leaf, 'Ateles')
    Node(140144824314704, leaf, 'Galago')
    In [*]: for node in r.postiter:
                print node # Nodes in postorder sequence.
    Node(140144824314576, leaf, 'Homo')
    Node(140144824314192, leaf, 'Pongo')
    Node(140144824314512, 'A')
    Node(140144824314256, leaf, 'Macaca')
    Node(140144824314448, 'B')
    Node(140144824314640, leaf, 'Ateles')
    Node(140144824314384, 'C')
    Node(140144824314704, leaf, 'Galago')
    Node(140144824314320, root, 'root')


We can access internal nodes using square brackets on the root node (or other
ancestor node). 

.. sourcecode:: ipython

    In [*]: r["C"] # You can use the node label
    Out[*]: Node(139624003155536, 'C')
    In [*]: r[139624003155536] # The node ID
    Out[*]: Node(139624003155536, 'C')
    In [*]: r[1] # Or the index of the node in preorder sequence
    Out[*]: Node(139624003155536, 'C')

We can access the information a node has about which other nodes it is
connected to using the `children` and `parent` attributes, which return
the nodes directly connected to the current node. The `descendants` method, on
the other hand, recursively lists all descendants of a node (not including 
the node itself)

.. sourcecode:: ipython

    In [*]: r["C"].children
    Out[*]: [Node(139624003155600, 'B'), Node(139624003155984, leaf, 'Ateles')]
    In [*]: r["B"].parent
    Out[*]: Node(139624003155536, 'C')
    In [*]: r["B"].descendants()
    Out[*]:
    [Node(139624003155664, 'A'),
     Node(139624003155792, leaf, 'Homo'),
     Node(139624003155856, leaf, 'Pongo'),
     Node(139624003155920, leaf, 'Macaca')]

We can search nodes using regular expressions with the Node grep method.
We can also grep leaf nodes and internal nodes specifically.

.. sourcecode:: ipython

    In [*]: r.grep("A") # By default, grep ignores case
    Out[*]: 
    [Node(139624003155664, 'A'),
     Node(139624003155920, leaf, 'Macaca'),
     Node(139624003155984, leaf, 'Ateles'),
     Node(139624003156048, leaf, 'Galago')]
    In [*]: r.grep("A", ignorecase = False)
    Out[*]: [Node(139624003155664, 'A'), Node(139624003155984, leaf, 'Ateles')
    In [*]: r.lgrep("A", ignorecase = False) # Limit our search to leaves
    Out[*]: [Node(140144824314640, leaf, 'Ateles')]
    In [*]: r.bgrep("Homo", ignorecase = False) # Limit our search to branches
    Out[*]: []






We can also search for nodes that match a certain criterion using the
`find` method. `find` takes a function that takes a node as its
first argument and returns a `bool`.

.. sourcecode:: ipython

    In [*]: def three_or_more_decs(node):
                return len(node) >= 4 
    In [*]: r.find(three_or_more_decs) # Find returns a generator
    Out[*]: <generator object find at 0x7efcbf498730>
    In [*]: r.findall(three_or_more_decs) # Findall returns a list
    Out[*]:
    [Node(139624003155728, root, 'root'),
     Node(139624003155536, 'C'),
     Node(139624003155600, 'B')]



Testing
=======

We can test many attributes of a node in Ivy.

We can test whether a node contains another node

.. sourcecode:: ipython

    In [*]: r["A"] in r["C"]
    Out[*]: True
    In [*]: r["C"] in r["A"]
    Out[*]: False
    In [*]: r["C"] in r["C"]
    Out[*]: True # Nodes contain themselves

We can test if a node is the root

.. sourcecode:: ipython

    In [*]: r.isroot
    Out[*]: True
    In [*]: r["C"].isroot
    Out[*]: False

We can test if a node is a leaf

.. sourcecode:: ipython

    In [*]: r.isleaf
    Out[*]: False
    In [*]: r["Homo"].isleaf
    Out[*]: True

We can test if a group of leaves is monophyletic

.. sourcecode:: ipython

    In [*]: r.ismono(r["Homo"], r["Pongo"])
    Out[*]: True
    In [*]: r.ismono(r["Homo"], r["Pongo"], r["Galago"])
    Out[*]: False

.. warning::
    `ismono` does not return an error if an internal node is given to it,
    but it does produce undesired results.

Modifying
=========

The Ivy Node object has many methods for modifying a tree in place.


Removing
--------

There are two main ways to remove nodes in Ivy; collapsing and pruning.

Collapsing removes a node and attaches its descendants to the node's parent.

.. sourcecode:: ipython

    In [*]: r["A"].collapse()
    In [*]: print r.ascii()
                                ------------+ Macaca 
                                :                    
                    -----------B+-----------+ Homo   
                    :           :                    
        -----------C+           ------------+ Pongo  
        :           :                                
    root+           ------------------------+ Ateles 
        :                                            
        ------------------------------------+ Galago 

Pruning removes a node and its descendants

.. sourcecode:: ipython

    In [*]: cladeB = r["B"] # Store this node: we will add it back later
    In [*]: r["B"].prune()
    In [*]: print r.ascii()
        -----------------C+-----------------+ Ateles 
    root+                                            
        ------------------------------------+ Galago

You can see that the tree now has a 'knee': clade C only has one child and
does not need to exist on the tree. We can remove it with another method of
removing nodes: excising. Excising removes a node from between its parent
and its single child.

.. sourcecode:: ipython

    In [*]: r["C"].excise()
    In [*]: print r.ascii()
        -------------------------------------+ Galago 
    root+                                             
        -------------------------------------+ Ateles 

It is important to note that although the tree has changed, the nodes in the
tree retain some of their original attributes, including their indices:

.. sourcecode:: ipython

    In [*]: r[0]
    Out[*]: Node(140144821291920, root, 'root')
    In [*]: r[1] # Node 1 ("C") no longer exists
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)

    IndexError: 1
    In [*]: r[7] # You can access existing nodes with their original indices
    Out[*]: Node(140144821292368, leaf, 'Ateles') 

To recap: 
#. `collapse` removes a node and adds its descendants to its parent
#. `prune` removes a node and also removes its descendants
#. `excise` removes 'knees'

Adding
------

Our tree is looking a little sparse, so let's add some nodes back in. There
are a few methods of adding nodes in Ivy. We will go over `biscect`, `add_child`, and `graft`

Bisecting creates a 'knee' node halfway between a parent and a child.

.. sourcecode:: ipython

    In [*]: r["Galago"].bisect_branch()
    Out[*]: Node(140144821654480)
    In [*]: print r.ascii
        ------------------------------------+ Ateles 
    root+                                            
        ------------------+-----------------+ Galago 

We now have a brand new node. We can set some of its attributes, including its
label.

Note: we `cannot` access this new node by using node indicies (that is, 
r[1], etc.). We also cannot use its label because it has none. We'll access
it using its ID instead (if you're following along, your ID will be different).

.. sourcecode:: ipython

    In [*]: r[140144821654480].label = "N"

Now let's add a node as a child of N. We can do this using the `add_child` method.

.. sourcecode:: ipython

    In [*]: r["N"].add_child(cladeB["Homo"])
    In [*]: print r.ascii()
        ------------------------------------+ Ateles 
    root+                                            
        :                 ------------------+ Galago 
        -----------------N+                          
                          ------------------+ Homo 

We can also add nodes with `graft`. `graft` adds a node as a sibling to the
current node. In doing so, it also adds a new node as parent to both nodes.

.. sourcecode:: ipython

    In [*]: r["Ateles"].graft(cladeB["Macaca"])
    In [*]: r["Galago"].graft(cladeB["Pongo"])
    In [*]: print r.ascii()
                    ------------------------+ Homo   
        -----------N+                                
        :           :           ------------+ Galago 
        :           ------------+                    
    root+                       ------------+ Pongo  
        :                                            
        :                       ------------+ Ateles 
        ------------------------+                    
                                ------------+ Macaca 


To recap:

#. `bisect_branch` adds 'knees'
#. `add_child` adds a node as a child to the current node
#. `graft` adds a node as a sister to the current node, and also adds a parent.


Ladderizing
-----------

Ladderizing non-destructively changes the tree so that it has a nicer-looking
output when drawn. It orders the clades by size.

.. sourcecode:: ipython

    In [*]: r.ladderize()
    In [*]: print r.ascii()
                                ------------+ Ateles 
        ------------------------+                    
        :                       ------------+ Macaca 
    root+                                            
        :           ------------------------+ Homo   
        -----------N+                                
                    :           ------------+ Galago 
                    ------------+                    
                                ------------+ Pongo  


Rerooting
---------

.. warning::
    Currently does not work properly.

.. sourcecode:: ipython

    In [*]: r.reroot(r["N"])
    In [*]: r.descendants() # Missing descendants
    Out[*]: 
    [Node(140144821839696),
     Node(140144821839120, leaf, 'Ateles'),
     Node(140144821839056, leaf, 'Macaca')]
    In [*]: print r.ascii() # Raises a KeyError

Writing
=======

Once you are done modifying your tree, you will probably want to save it.
You can save your trees with the `write` function. This function
takes a root node and an open file object as inputs. This function can
currently only write in newick format.


.. sourcecode:: ipython

    In [*]: f = open("examples/primates_mangled.newick", "w")
    In [*]: ivy.tree.write(r, outfile = f)
    In [*]: f.close()













