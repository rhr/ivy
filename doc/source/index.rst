*************************************
ivy: interactive visual phylogenetics
*************************************

``ivy`` is a Python module for the analysis and exploration of
phylogenetic trees and comparative data, based on IPython, matplotlib,
scipy, and numpy. 

.. warning::

   This project is in very early stages of development, and has bugs,
   an unstable API, and incomplete documentation.  Address questions
   to Rick Ree <rree@fieldmuseum.org>.

.. toctree::
   :maxdepth: 2

   Downloading and Installing <install>
   User's Guide <using>
   API Documentation <modules>

Motivation
==========

* Phylogenetics and comparative analysis is fun and rewarding.

* Simple and common tasks should be easy, and complex tasks should be
  possible.

* Data *exploration* is aided by interactive visual tools, but data
  *analysis* is better accomplished by scripts that can be reused and
  modified.

Objectives
==========

To provide the following components for users:

* A hybrid user interface consisting of 

  #. a **magic command prompt** providing dynamic autocompletion and
     on-the-fly Python programming, and

  #. **interactive visual tools** for exploring phylogenetic trees,
     data, and comparative methods.

* A clean library API making it easy to create

  #. **custom scripts** for data analysis

  #. **custom modules** implementing comparative methods and
     visualization tools.

* Wrapper modules for additional libraries such as Biopython and
  DendroPy, including support for R packages (e.g. diversitree,
  geiger, laser, etc.) using RPy.

* Examples and tutorials (e.g., using IPython's interactive ``demo``
  module) for teaching phylogenetics.

Design principles
=================

* Good user interfaces are discoverable, intuitive, and predictable.

* Well-documented and self-documenting code makes learning and
  extending an API easier.

* Readable code is better than fast code (most of the time).

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
