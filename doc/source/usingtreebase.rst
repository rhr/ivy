**************
Using Treebase
**************

``ivy`` has functions to pull trees from `Treebase <http://treebase.org/treebase-web/about.html;jsessionid=5B7D6A265E17EFAB9565327D3A78CD4B>`_.


Fetching the study
==================

If you have an id for a study on treebase, you can fetch the study and access the trees contained within the study.

.. sourcecode:: ipython

    In [1]: import ivy
    In [2]: from ivy.treebase import fetch_study
    In [3]: study_id = "1411" # The leafy cactus genus Pereskia
    In [4]: e = fetch_study(study_id, 'nexml') # e is an lxml etree


Accessing the tree
==================

You can parse the output of fetch_study using the parse_nexml function, then access the tree(s) contained within the study.

.. sourcecode:: ipython

    In [5]: from ivy.treebase import parse_nexml
    In [6]: x = parse_nexml(e) # x is an ivy Storage object
    In [7]: r = x.trees[0].root
    In [8]: from ivy.interactive import treefig
    In [9]: fig = treefig(r)
