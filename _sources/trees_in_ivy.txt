Trees in Ivy
=============

``ivy`` does not have a tree class per se; rather trees in ``ivy`` exist as collections
of nodes. In ``ivy``, a ``Node`` is a class that contains information about a node.
``Nodes`` are rooted and recursively contain their children. Functions
in ``ivy`` act directly on ``Node`` objects. Nodes support Python idioms such as ``in``,
``[``, iteration, etc. This guide will cover how to read, view, navigate, modify,
and write trees in ``ivy``.

Reading
-------

You can read in trees using ``ivy``'s ``tree.read`` function. This function supports
newick and nexus files. The ``tree.read`` function can take a file name, a file
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

You can copy a read tree using the ``copy`` method on the root node. ``Node``
objects are mutable, so this method is preferred over ``r2 = r`` if you want
to create a deep copy.

.. sourcecode:: ipython

    In [*]: r2 = r.copy()

Viewing
-------

There are a number of ways you can view trees in ``ivy``. For a simple display
without needing to create a plot, ``ivy`` can create ascii trees that can be
printed to the console.

.. sourcecode:: ipython

    In [*]: print r.ascii() # You can use the ascii method on root nodes.
                                   ---------+ Homo
                          --------A+
                 --------B+        ---------+ Pongo
                 :        :
        --------C+        ------------------+ Macaca
        :        :
    root+        ---------------------------+ Ateles
        :
        ------------------------------------+ Galago

For a more detailed and interactive tree, ``ivy`` can create a plot using
``matplotlib``. More detail about visualization using ``matplotlib`` are in the
"Visualization with matplotlib" section.

.. sourcecode:: ipython

    In [*]: import ivy.vis
    In [*]: fig = ivy.vis.treevis.TreeFigure(r)
    In [*]: fig.show()

.. image:: _images/primate_mpl.png
    :width: 700


You can also create a plot using ``Bokeh``.

.. sourcecode:: ipython

    In [*]: import ivy.vis.bokehtree
    In [*]: fig2 = ivy.vis.bokehtree.BokehTree(r)
    In [*]: fig2.drawtree()

.. image:: _images/primate_bokeh.png
    :width: 700


Navigating
----------

A node in ``ivy`` is a container. It recursively contains its descendants,
as well as itself. You can navigate a tree using the Python idioms that
you are used to using.

Let's start by iterating over all of the children contained within the root
node. By default, iteration over a node happens in preorder sequence, starting
with the root node. To iterate over a node in a specific sequence, you can use
the ``preorder`` and ``postorder`` methods.

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
    In [*]: for node in r.preiter():
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
    In [*]: for node in r.postiter():
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
connected to using the ``children`` and ``parent`` attributes, which return
the nodes directly connected to the current node. The ``descendants`` method, on
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

We can search nodes using regular expressions with the ``Node`` ``grep`` method.
We can also ``grep`` leaf nodes and internal nodes specifically.

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
``find`` method. ``find`` takes a function that takes a node as its
first argument and returns a ``bool``.

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
-------

We can test many attributes of a node in ``ivy``.

We can test whether a node contains another node. Recall that a node contains
all of its descendants as well as itself.

.. sourcecode:: ipython

    In [*]: r["A"] in r["C"]
    Out[*]: True # Nodes contain descendants
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

Modifying
---------

The ``ivy`` ``Node`` object has many methods for modifying a tree in place.


Removing
~~~~~~~~

There are two main ways to remove nodes in ``ivy``; collapsing and pruning.

Collapsing removes a node and attaches its descendants to the node's parent.

.. sourcecode:: ipython

    In [*]: r["A"].collapse()
    Out[*]: Node(140622783265744, 'B') # This function returns the node's parent and also alters the tree
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

    In [*]: r["B"].prune()
    In [*]: print r.ascii()
        -----------------C+-----------------+ Ateles
    root+
        ------------------------------------+ Galago

You can see that the tree now has a 'knee': clade C only has one child and
does not need to exist on the tree. We can remove it with another method of
removing nodes: ``excise``. Excising removes a node from between its parent
and its single child.

.. sourcecode:: ipython

    In [*]: r["C"].excise()
    In [*]: print r.ascii()
        -------------------------------------+ Galago
    root+
        -------------------------------------+ Ateles


To recap:

#. ``collapse`` removes a node and adds its descendants to its parent
#. ``prune`` removes a node and also removes its descendants
#. ``excise`` removes 'knees'

Adding
~~~~~~

Our tree is looking a little sparse, so let's add some nodes back in. There
are a few methods of adding nodes in ``ivy``. We will go over ``biscect``,
``add_child``, and ``graft``

Bisecting creates a 'knee' node halfway between a parent and a child.

.. sourcecode:: ipython

    In [*]: r["Galago"].bisect_branch()
    Out[*]: Node(140144821654480)
    In [*]: print r.ascii()
        ------------------------------------+ Ateles
    root+
        ------------------+-----------------+ Galago

We now have a brand new node. We can set some of its attributes, including its
label.

Note: modifying a tree can have unwanted side effects for node indices.
Watch what happens when we print out the pre-order index for each node:

.. sourcecode:: ipython

    In [*]: print [n.ni for n in r]
    [0, 7, None, 8]

We would expect the indices to be [0,1,2,3]. We can fix the indices by calling
the ```reindex`` method.

.. sourcecode:: ipython

    In [*] r.reindex()
    In [*] print [n.ni for n in r]
    [0, 1, 2, 3]

Now that we have fixed the indices, we can access the new node by its index and
set its label.


.. sourcecode:: ipython

    In [*]: r[2].label = "N"

Now let's add a node as a child of N. We can do this using the ``add_child``
method. Let's use a node from the copy of ``r`` we made, ``r2``.

.. sourcecode:: ipython

    In [*]: r["N"].add_child(r2["Homo"])
    In [*]: print r.ascii()
        ------------------------------------+ Ateles
    root+
        :                 ------------------+ Galago
        -----------------N+
                          ------------------+ Homo

We can also add nodes with ``graft``. ``graft`` adds a node as a sibling to the
current node. In doing so, it also adds a new node as parent to both nodes.

.. sourcecode:: ipython

    In [*]: r["Ateles"].graft(r2["Macaca"])
    In [*]: r["Galago"].graft(r2["Pongo"])
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

#. ``bisect_branch`` adds 'knees'
#. ``add_child`` adds a node as a child to the current node
#. ``graft`` adds a node as a sister to the current node, and also adds a parent.


Ladderizing
~~~~~~~~~~~

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
~~~~~~~~~

.. warning::
    This reroot function has not been thouroughly tested. Use with caution.

All trees in ``ivy`` are rooted. If you read in a tree that has been incorrectly
rooted, you may want to reroot it. You can do this with the ``reroot``
function. This function returns the root node of the rerooted tree. Note that
unlike previous functions, the reroot function returns a *new* tree. The
old tree is not modified.

.. sourcecode:: ipython

    In [*]: r_reroot = r.reroot(r["Galago"])
    In [*]: print r_reroot.ascii()
    ----------------------------------------+ Galago
    +
    :         ------------------------------+ Pongo
    ----------+
              :         --------------------+ Homo
              ---------N+
                        :         ----------+ Ateles
                        ----------+
                                  ----------+ Macaca

Dropping Tips
~~~~~~~~~~~~~

You can remove leaf nodes with the ``drop_tips`` function. Note that
this function returns a *new* tree. The old tree is not modified.
This function takes a list of tip labels as input.


.. sourcecode:: ipython

    In [*]: r_dropped = r_reroot.drop_tip(["Pongo", "Macaca"])

Writing
-------

Once you are done modifying your tree, you will probably want to save it.
You can save your trees with the ``write`` function. This function
takes a root node and an open file object as inputs. This function can
currently only write in newick format.


.. sourcecode:: ipython

    In [*]: with open("primates_altered.newick", "w") as f:
                ivy.tree.write(r_dropped, outfile = f)


