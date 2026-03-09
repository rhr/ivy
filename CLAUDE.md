# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`ivy` is a Python library for interactive visual phylogenetics, built on numpy, scipy, matplotlib, and IPython. It has no Tree class — trees are represented as linked `Node` objects, and most functions operate directly on nodes.

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ivy

# Install in development mode
pip install -e .
```

Or manually install dependencies: `ipython numpy scipy matplotlib pandas biopython pillow pyparsing lxml`

## No Test Suite

There is no automated test suite. Manual testing is done interactively via IPython:

```python
from ivy.interactive import *
root = readtree("examples/plants.newick")  # or any newick file/string
fig = treefig(root)
```

## Architecture

### Core Data Model (`ivy/tree.py`)
- `Node` class: the fundamental unit. Nodes have `parent`, `children`, `label`, `length`, `age`, `isleaf`, `isroot`, etc.
- Trees are rooted `Node` graphs — pass the root node to tree functions.
- `tree.read(data)` parses newick strings, file paths, or file objects and returns the root `Node`.

### Parsing (`ivy/newick.py`, `ivy/nexus.py`)
- `newick.parse(s)` — tokenizes and parses newick format
- `nexus` — handles NEXUS format files

### Layout (`ivy/layout.py`, `ivy/layout_polar.py`)
- `cartesian()` — computes (x, y) coordinates for nodes for rectangular layout
- `layout_polar` — polar/radial layout variant

### Visualization (`ivy/vis/`)
- `ivy/vis/tree.py` — `TreeFigure` and `MultiTreeFigure`: interactive matplotlib windows for tree display with pan/zoom, node selection, overview pane
- `ivy/vis/alignment.py` — `AlignmentFigure` for sequence alignments
- `ivy/vis/bokehtree.py` — Bokeh-based tree viewer (optional, bokeh not in default requirements)
- `ivy/vis/__init__.py` — re-exports `TreeFigure`, `MultiTreeFigure`, `AlignmentFigure`, `JuxtaposerFigure`

### Interactive Entry Point (`ivy/interactive.py`)
- `readtree(data)` — wraps `tree.read()`
- `treefig(root)` — creates and shows a `TreeFigure`
- Designed for `from ivy.interactive import *` in IPython sessions

### Analysis Modules
- `ivy/contrasts.py` — phylogenetic independent contrasts
- `ivy/ages.py` — divergence time utilities
- `ivy/birthdeath.py` — birth-death models
- `ivy/chars/` — character evolution (`mk.py` = Mk model, `catpars.py`, `evolve.py`)
- `ivy/ltt.py` — lineages-through-time
- `ivy/bipart.py` — bipartitions

### Data I/O
- `ivy/genbank.py` — GenBank sequence fetching
- `ivy/align.py`, `ivy/sequtil.py` — sequence alignment utilities
- `ivy/treebase.py`, `ivy/ubio.py` — TreeBASE/uBio web services (require lxml)

### Storage (`ivy/storage.py`)
- `Storage` class used by `TreeFigure` for node data caching
