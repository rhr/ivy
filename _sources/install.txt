.. _getting-ivy

**************************
Downloading and Installing
**************************

Ivy is a Python package, and can be as simple to install as:

.. code-block:: bash

  $ pip install git+git://github.com/rhr/ivy.git

However, ``ivy`` requires quite a bit of third-party open source
software to work.  The following instructions assume a Debian-based
Linux system like Ubuntu.  On a Mac, you can `use Anaconda <http://docs.continuum.io/anaconda/install>`_ to install
dependencies.

More detailed instructions for Mac and Windows are in the works.

Install Guide
=============

Windows
~~~~~~~

To install ``ivy`` on Windows, you must first install a few dependencies.

First, you must have Python 2.7 installed. ``ivy`` is currently not
compatible with Python 3.

The easiest way to install ``ivy`` and its dependencies is to use ``pip``.
Python 2.7.9+ is shipped with ``pip``. If you have an earlier version of python,
you must install ``pip``. Instructions can be found here: `How to install pip
on Windows <http://stackoverflow.com/questions/4750806/how-to-install-pip-on-windows>`

You may need to add the path to ``pip`` to your PATH variable. If you have
a newer version of python, ``pip`` will be automatically installed into
``C:\Python27\Scripts\pip``. To add this to your PATH variable, run the
following:

.. code-block:: bash

    setx PATH "%PATH%;C:\Python27\Scripts"

Once ``pip`` is installed, dependencies can be installed as follows:

First, install Microsoft Visual C++ Compiler for Python 2.7 if you do not
have it already: http://www.microsoft.com/en-us/download/details.aspx?id=44266

Then, install the package dependencies
.. code-block:: bash
    pip install matplotlib :: This will also install numpy
    pip install biopython
    pip install pyparsing
    pip install lxml
    pip install bokeh
    pip install pydf


Next you need to install SciPy. It may be easiest to download the binary from here:
http://www.lfd.uci.edu/~gohlke/pythonlibs/. Look for either
scipy‑0.16.0‑cp27‑none‑win32.whl or scipy‑0.16.0‑cp27‑none‑win_amd64.whl, depending
on whether you have 32- or 64-bit python. Then run:

..code-block::bash
    pip install /path/to/binary/scipy‑0.16.0‑cp27‑none‑win32.whl


It is recommended that you run ``ivy`` using ``ipython``.

..code-block::bash
    pip install ipython

It is also recommended that you run ``ivy`` in a VirtualEnvironment

..code-block::bash
    pip install virtualenv :: install virtualenv
    virtualenv mypy :: Create the virtualenvironment
    mypy\Scripts\activate :: Run the virutalenvironment

Now you may install ``ivy``

..code-block::bash
    pip install git+git://github.com/rhr/ivy.git@christie-master



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
* `bokeh <http://bokeh.pydata.org/en/latest/>`_ - visualization

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

  $ pip install git+git://github.com/rhr/ivy.git


Source code
===========

Ivy source code is hosted at https://github.com/rhr/ivy
