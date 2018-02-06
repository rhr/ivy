************
Installation
************

``Ivy`` is a module for Python 3. You probably want to install it into
a `virtual environment`_. A convenient way to do this is using conda_,
the package/environment manager provided by the Anaconda_ Python
distribution and its minimalist cousin, Miniconda_.

.. _virtual environment: https://docs.python.org/3/tutorial/venv.html
.. _conda: https://conda.io/docs
.. _Anaconda: https://www.anaconda.com/distribution
.. _Miniconda: https://conda.io/miniconda.html

Installation using conda_
=========================

This is the recommended way, unless you prefer another system of
managing your environments, in which case adapting the instructions
below should be straightforward.

Install Anaconda_ or Miniconda_
-------------------------------

Install the **Python 3** version of either Anaconda_ (large, slow to
download, includes most dependencies of ``ivy``) or Miniconda_ (small,
fast to download, includes no dependencies).

The following assumes you now have ``conda`` in your ``$PATH`` and
you're using a Mac or Linux terminal. The same tasks can be done using
the Anaconda Navigator's graphical interface.

Create a new environment for `ivy`
----------------------------------

This is optional, but probably a good idea.

.. code-block:: bash

  $ conda create -n ivy  # here 'ivy' is just the environment name

Activate the environment. The environment's name should appear in the
shell prompt.

.. code-block:: bash

  $ source activate ivy
  (ivy) $

Install dependencies
--------------------

``Ivy`` liberally draws from the amazing ecosystem of Python data
science libraries, such as scipy_, pandas_, and especially matplotlib_.

.. _scipy: https://scipy.org
.. _pandas: https://pandas.pydata.org
.. _matplotlib: https://matplotlib.org

The advantage of using ``conda`` here is that it pulls in compiled
versions of the packages needed by ``ivy``, and all their
dependencies; you could use ``pip`` instead, but depending on your
setup, it may require compiling things that you're not prepared to
compile. (With the exception of ``biopython``, all of ``ivy``'s
dependencies are already included in Anaconda.)

.. code-block:: bash

  (ivy) $ conda install numpy scipy matplotlib pandas biopython pyparsing lxml jupyter

This will pull in many other dependencies, and may take several
minutes to complete.

Finally install ``ivy``:

.. code-block:: bash

  (ivy) $ pip install git+http://github.com/rhr/ivy.git

This pulls the latest source code from the `github repository
<https://github.com/rhr/ivy>`_.

Updating ``ivy``
================

First remove the existing installation:

.. code-block:: bash

  (ivy) $ pip uninstall ivy-phylo

Then re-install it:

.. code-block:: bash

  (ivy) $ pip install git+http://github.com/rhr/ivy.git
