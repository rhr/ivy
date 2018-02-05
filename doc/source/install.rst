.. _getting-ivy

************
Installation
************

``Ivy`` is a module for Python 3. You probably want to install it into
a virtual environment. A convenient way to do this is via `miniconda
<https://conda.io/miniconda.html>`_.

Miniconda
=========

Once ``miniconda`` is installed, you can create a new conda
environment for ``ivy`` like so:

.. code-block:: bash

  $ conda create -n ivy

Activate the environment:

.. code-block:: bash

  $ source activate ivy
  (ivy) $

Next, install the dependencies:

* `numpy <http://numpy.scipy.org>`_
* `scipy <http://www.scipy.org>`_
* `matplotlib <http://matplotlib.org>`_
* `pandas <http://pandas.pydata.org>`_
* `biopython <http://www.biopython.org>`_
* `pyparsing <http://pyparsing.wikispaces.com>`_
* `lxml <http://lxml.de>`_

And, for interactive use:

* `jupyter <http://jupyter.org>`_

.. code-block:: bash

  (ivy) $ conda install numpy scipy matplotlib pandas biopython pyparsing lxml jupyter

This will pull in many other dependencies, and may take several
minutes to complete.

Finally install ``ivy``:

.. code-block:: bash

  (ivy) $ pip install git+http://github.com/rhr/ivy.git
		

Ubuntu + virtualenv
===================

On Ubuntu you can install ``ivy``'s dependencies via system
packages. (If Python 3 is your system's default, replace python3 with
python below.)

.. code-block:: bash

  $ sudo apt install python3-matplotlib python3-scipy python3-numpy python3-biopython python3-pyparsing python3-pillow python3-lxml python3-pandas

System packages are often out of date, but not to worry, you can
install everything you need to compile new versions for installation
into a virtual environment.

.. code-block:: bash

  $ sudo apt build-dep python3-matplotlib python3-scipy python3-numpy python3-biopython python3-pyparsing python3-pillow python3-lxml python3-pandas

Preparing a virtual Python environment
======================================

`virtualenv <http://pypi.python.org/pypi/virtualenv>`_ allows you to
create sandboxed Python environments in which it is safe to install
bleeding-edge third-party modules without touching any system
files.

.. code-block:: bash

  $ sudo apt install python3-virtualenv python3-pip

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

  $ for module in matplotlib scipy numpy pandas biopython pyparsing lxml ; do
  $    pip install $module ;
  $ done

Jupyter
-------

You will also want to install Jupyter in your virtual environment:

.. code-block:: bash

  $ pip install jupyter

Installing ``ivy``
==================

Finally, once the dependencies have been satisfied, we can install ``ivy``:

.. code-block:: bash

  $ pip install git+http://github.com/rhr/ivy


Source code
===========

Ivy source code is hosted at https://github.com/rhr/ivy

