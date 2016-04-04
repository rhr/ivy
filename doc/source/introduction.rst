
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

You can copy a read tree using the ``copy_new`` method on the root node. ``Node``
objects are mutable, so this method is preferred over ``r2 = r`` if you want
to create a deep copy.

.. sourcecode:: ipython

    In [*]: r2 = r.copy_new(recurse=True) # If recurse=False, won't copy children.

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

Modifying
---------

The ``ivy`` ``Node`` object has many methods for modifying a tree in place.


Removing
~~~~~~~~

There are two main ways to remove nodes in ``ivy``; collapsing and pruning.

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
removing nodes: ``excise``. Excising removes a node from between its parent
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

Now let's add a node as a child of N. We can do this using the ``add_child``
method.

.. sourcecode:: ipython

    In [*]: r["N"].add_child(cladeB["Homo"])
    In [*]: print r.ascii()
        ------------------------------------+ Ateles
    root+
        :                 ------------------+ Galago
        -----------------N+
                          ------------------+ Homo

We can also add nodes with ``graft``. ``graft`` adds a node as a sibling to the
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

    In [*]: r2 = r.reroot(r["Galago"])
    In [*]: print r2.ascii()
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

    In [*]: r3 = r.drop_tips(["Pongo", "Macaca"])

Writing
-------

Once you are done modifying your tree, you will probably want to save it.
You can save your trees with the ``write`` function. This function
takes a root node and an open file object as inputs. This function can
currently only write in newick format.


.. sourcecode:: ipython

    In [*]: with open("examples/primates_mangled.newick", "w") as f:
                ivy.tree.write(r3, outfile = f)



Using Treebase
==============

``ivy`` has functions to pull trees from `Treebase <http://treebase.org/treebase-web/about.html;jsessionid=5B7D6A265E17EFAB9565327D3A78CD4B>`_.


Fetching the study
------------------

If you have an id for a study on treebase, you can fetch the study and
access the trees contained within the study.

.. sourcecode:: ipython

    In [*]: import ivy
    In [*]: from ivy.treebase import fetch_study
    In [*]: study_id = "1411" # The leafy cactus genus Pereskia
    In [*]: e = fetch_study(study_id, 'nexml') # e is an lxml etree


Accessing the tree
------------------

You can parse the output of fetch_study using the parse_nexml function,
 then access the tree(s) contained within the study.

.. sourcecode:: ipython

    In [*]: from ivy.treebase import parse_nexml
    In [*]: x = parse_nexml(e) # x is an ivy Storage object
    In [*]: r = x.trees[0].root
    In [*]: from ivy.interactive import treefig
    In [*]: fig = treefig(r)



Visualization with Matplotlib
=============================

``ivy`` supports interactive tree visualization with Matplotlib.

Small Trees
-----------

Displaying a tree is very simple. For interactive tree viewing, you can run
the command ``from ivy.interactive import *``, which imports a number of
convenience functions for interacting with trees. After importing everything
from ivy.interactive, you may, for instance, use ``readtree`` instead of
``ivy.tree.read`` and ``treefig`` instead of ``ivy.vis.tree.TreeFigure``.

.. sourcecode:: ipython

    In [*]: from ivy.interactive import *
    In [*]: r = readtree("examples/primates.newick")
    In [*]: fig = treefig(r)

You can also use the magic command ``%maketree`` in the ipython console to
read in a tree.

.. sourcecode:: ipython

    In [*]: %maketree
    Enter the name of a tree file or a newick string:
    examples/primates.newick
    Tree parsed and assigned to variable 'root'
    In [*]: root
    Out[*]: Node(139904996110480, root, 'root')


.. image:: _images/visualization_1.png
    :width: 700

A tree figure by default consists of the tree with clade and leaf
labels and a navigation toolbar. The navigation toolbar allows zooming and
panning. Panning can be done by clicking with the middle mouse button, using
the arrow keys, or using the pan tool on the toolbar. Zooming can be done
using the scroll wheel, the plus and minus keys, or the 'zoom to rectangle'
tool in the toolbar. Press t to return default zoom level.

Larger trees are shown with a split overview pane as well, which can be toggled
with the ``toggle_overview`` method.

.. sourcecode:: ipython

    In [*]: fig.toggle_overview()

.. image:: _images/visualization_2.png
    :width: 700

You can retrieve information about a node or group of nodes by selecting
them (selected nodes have green circles on them)
and accessing the ``selected`` nodes

.. sourcecode:: ipython

    In [*]: fig.selected
    Out [*]:
    [Node(139976891981456, leaf, 'Homo'),
     Node(139976891981392, 'A'),
     Node(139976891981520, leaf, 'Pongo')]

.. image:: _images/visualization_3.png
    :width: 700


You can also select nodes from the command line. Entering an internal node will
select that node and all of its descendants.

.. sourcecode:: ipython

    In [*]: fig.select_nodes(r["C"])

.. image:: _images/visualization_4.png
    :width: 700

You can highlight certain branches using the ``highlight`` method. Again,
entering an internal node will highlight that node and its descendants.

You can optionally show the highlighted branches on the overview panel using
the ``ov`` keyword

.. sourcecode:: ipython

    In [*]: fig.highlight(r["B"], ov=True)

.. image:: _images/visualization_5.png
    :width: 700

You can add layers of various kinds using the ``add_layers`` method. The
``layers`` module contains various functions for adding layers to the tree,
including images, labels, shapes, etc.

In fact, the ``highlight`` method is simply a wrapper for an ``add_layers``
call.

.. sourcecode:: ipython

    In [*]: from ivy.vis import layers
    In [*]: fig.redraw() # This clears the plot
    In [*]: fig.add_layer(layers.add_circles, r.leaves(),
            colors = ["red", "orange", "yellow", "green", "blue"],
            ov=False) # Prevent layer from appearing on overview with ov keyword

.. image:: _images/visualization_6.png
    :width: 700

The new layer will be cleared with the next call to ``fig.redraw``. You can
store a layer and draw it every time using the ``store`` keyword. We can
access our stored layers through the ``layers`` attribute of the figure

.. sourcecode:: ipython
    In [*]: fig.add_layer(layers.add_circles, r.leaves(),
            colors = ["red", "orange", "yellow", "green", "blue"],
            ov=False, store="circles")
    In [*]: fig.layers
    Out[*]: OrderedDict([('leaflabels', <functools.partial object at 0x7feda07292b8>), ('branchlabels', <functools.partial object at 0x7feda084b100>), ('circles', <functools.partial object at 0x7feda0752af8>)])

As we can see, our figure has "leaflabels" and "branchlabels" as layers, as
well as the new "circles" layer. You can toggle the visibility of a layer
using the ``toggle_layer`` method and the layer's name. The layer is still
there and can be accessed with ``fig.layers``, but it is not visible on
the plot.

.. sourcecode:: ipython
    In [*]: fig.toggle_layer("circles")

Large Trees
-----------

Oftentimes, the tree you are working with is too large to comfortably fit on
one page. ``ivy`` has many tools for working with large trees and creating
legible, printable figures of them. Let's try working on the plant phylogeny.

.. sourcecode:: ipython

    In [*]: r = readtree("examples/plants.newick")
    In [*]: fig = treefig(r)

.. image:: _images/plants_fig1.png
    :width: 700

When a tree has a large number of tips (>100), ``ivy`` automatically includes an
overview on the side. This tree looks rather cluttered. We can try to clean it
up by ladderizing the tree and toggling off the node labels

.. sourcecode:: ipython

    In [*]: fig.ladderize()
    In [*]: fig.toggle_branchlabels()

.. image:: _images/plants_fig2.png
    :width: 700

Here you can see that when all of the tip labels do not fit on the tree, the
plot automatically only draws as many labels as will fit.

Let's say we only want to look at the Solanales. The ``highlight`` function,
combined with the ``find`` function, is very useful when working with large
trees.

.. sourcecode:: ipython

    In [*]: sol = fig.find("Solanales")[0]
    In [*]: fig.highlight(sol)

.. image:: _images/plants_fig3.png
    :width: 700

We can zoom to this clade with the ``zoom_clade`` function.

.. sourcecode:: ipython

    In [*]: fig.zoom_clade(sol)

.. image:: _images/plants_fig4.png
    :width: 700

Maybe we want to zoom out a little. We can select a few clades...

.. image:: _images/plants_fig5.png
    :width: 700

And then zoom to the MRCA of the selected nodes

.. sourcecode:: ipython

    In [*]: c = fig.root.mrca(fig.selected)
    In [*]: fig.zoom_clade(c)

.. image:: _images/plants_fig6.png
    :width: 700

Another benefit to using ``ivy`` interactively is ``ivy``'s node autocomplete
function. You can type in the partial name of a node and hit ``tab`` to
autocomplete, just like with any other autocompletion in ipython.

.. sourcecode:: ipython

    In [*]: fig.root["Sy # Hit tab to autocomplete
    Sylvichadsia  Symplocaceae  Synoum        Syrmatium
    In [*]: fig.root["Sym # Hitting tab will complete the line
    In [*]: fig.root["Symplocaceae"]
    Out[*]: Node(139904995827408, leaf, 'Symplocaceae')

``ivy`` also has tools for printing large figures across multiple pages. The
figure method ``hardcopy`` creates an object that has methods for creating
PDFs that can be printed or placed in documents. To print a large figure
across multiple pages, you can use the ``render_multipage`` method of a
``hardcopy`` object. For more information, look at the documentation for
``render_multipage``. The following code will create a PDF that has the figure
spread across 4x4 letter-size pages.

.. sourcecode:: ipython

    In [*]: h = fig.hardcopy()
    In [*]: h.render_multipage(outfile="plants.pdf", dims = [34.0, 44.4])

