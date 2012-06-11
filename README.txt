=====================================
ivy: interactive visual phylogenetics
=====================================

``ivy`` is a lightweight library and an interactive visual programming
environment for phylogenetics.  It is built on a powerful foundation
of open-source software components, including numpy, scipy,
matplotlib, and IPython.

``ivy`` emphasizes interactive, exploratory visualization of
phylogenetic trees.  For example::

    #!/usr/bin/env ipython
    from ivy.interactive import *
    
    root = readtree("primates.newick")
    fig = treefig(root)


Documentation and other resources
=================================

http://www.reelab.net/ivy


Installation
============

0. Install `virtualenv <http://pypi.python.org/pypi/virtualenv>`_.

1. Install ivy: ``pip install ivy``



