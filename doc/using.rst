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

Starting the shell
==================

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
==============

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

