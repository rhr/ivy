# ivy: interactive visual phylogenetics

`ivy` is a lightweight library and an interactive visual programming
environment for phylogenetics.  It is built on a powerful foundation
of open-source software components, including numpy, scipy,
matplotlib, and IPython.

`ivy` emphasizes interactive, exploratory visualization of
phylogenetic trees.  For example, in IPython:


```python
from ivy.interactive import *

root = readtree("primates.newick")
fig = treefig(root)
```

This will open an interactive tree window that allows panning,
zooming, and node selection with the mouse. Additionally, you can
manipulate the figure from the console, e.g.:

```python
fig.ladderize()
fig.toggle_leaflabels()
```

## Documentation and other resources

(Woefully out of date and incomplete)

http://www.reelab.net/ivy

## Installation


Recommended for non-developers: use [conda](https://conda.io/miniconda.html) for Python 3.

1. Download and install Miniconda 3 for Python 3

2. Create an environment for `ivy`:

  ```bash
  conda create -n ivy ipython jupyter numpy scipy matplotlib biopython pillow pyparsing lxml
  ```
  
3. Activate the environment:

  ```bash
  source activate ivy
  ```

3. Install `ivy`:

  ```bash
  pip install https://github.com/rhr/ivy/zipball/master
  ```
  
4. To update, uninstall it first:
  ```bash
  pip uninstall ivy-phylo
  ```
  then re-run `pip` as above.
