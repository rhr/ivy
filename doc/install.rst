.. _getting-ivy

**************************
Downloading and Installing
**************************

Ivy is a Python package, and can be as simple to install as:

.. code-block:: bash

  $ pip install ivy-phylo

However, ``ivy`` requires quite a bit of third-party open source
software to work.  The following instructions assume a Debian-based
Linux system like Ubuntu.  On a Mac, they should work if you've
installed the Apple Developer Tools - just ignore the ``apt-get``
lines, and instead use ``easy_install`` and ``pip``.

More detailed instructions for Mac and Windows are in the works.

Dependencies
============

``ivy`` depends on several Python libraries for numerical and other
kinds of specialized functions.

* `matplotlib <http://matplotlib.sf.net>`_ (>=1.0) - cross-platform,
  toolkit-independent graphics for interactive visualization
* `scipy <http://www.scipy.org>`_ - high-level scientific modules for
  statistics, optimization, etc.
* `numpy <http://numpy.scipy.org>`_ - fast numerical functions for
  N-dimensional arrays
* `biopython <http://www.biopython.org>`_ - for handling molecular
  sequences: converting between formats, querying and retrieving data
  from GenBank, etc.
* `pyparsing <http://pyparsing.wikispaces.com>`_ - convenience
  functions for parsing text

These are easily installed by:

.. code-block:: bash

  $ sudo apt-get install python-matplotlib python-scipy python-numpy python-biopython python-pyparsing

However, the precompiled packages available for your system may not be
up to date - in particular, your distribution may not provide
matplotlib 1.0 or higher.  In which case you are better off compiling
your own in a virtual Python environment using `virtualenv
<http://pypi.python.org/pypi/virtualenv>`_ and `pip
<http://www.pip-installer.org>`_.

Before proceeding, let's make sure we have everything we need to
compile the modules:

.. code-block:: bash

  $ sudo apt-get build-dep python-matplotlib python-scipy python-numpy python-biopython python-pyparsing

Preparing a virtual Python environment
======================================

`virtualenv <http://pypi.python.org/pypi/virtualenv>`_ allows you to
create sandboxed Python environments in which it is safe to install
bleeding-edge third-party modules without touching any system
files.

.. code-block:: bash

  $ sudo apt-get install python-virtualenv

or

.. code-block:: bash

  $ sudo easy_install virtualenv

`pip <http://www.pip-installer.org>`_ is an improved replacement of
``easy_install``, and can be installed by:

.. code-block:: bash

  $ sudo apt-get install python-pip

or

.. code-block:: bash

  $ sudo easy_install pip

The next step is to create a virtual Python environment:

.. code-block:: bash

  $ virtualenv mypy

where ``mypy`` is an arbitrary name.  The environment can be activated
by 'sourcing' the ``activate`` script:

.. code-block:: bash

  $ . mypy/bin/activate

To make it your default Python environment, simply prepend
``$HOME/mypy/bin`` to your ``PATH``, e.g., in your ``.bashrc`` file:

.. code-block:: bash

  $ export PATH=$HOME/mypy/bin:$PATH

Once the environment is active, we can install the modules themselves:

.. code-block:: bash

  $ for module in matplotlib scipy numpy biopython pyparsing ; do
  $    pip install $module ;
  $ done

IPython
-------

You will also want to install IPython in your virtual environment:

.. code-block:: bash

  $ pip install ipython

Installing ``ivy``
==================

Finally, once the dependencies have been satisfied, we can install ``ivy``:

.. code-block:: bash

  $ pip install ivy-phylo


Source code
===========

Ivy source code is hosted at http://launchpad.net/ivy and can be
checked out via bazaar:

.. code-block:: bash

  $ bzr branch lp:ivy

