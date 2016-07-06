"""
ivy - a phylogenetics library and visual shell
http://www.reelab.net/ivy

Copyright 2010 Richard Ree <rree@fieldmuseum.org>

Required: ipython, matplotlib, scipy, numpy
Useful: dendropy, biopython, etc.
"""
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.

## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program. If not, see
## <http://www.gnu.org/licenses/>.
from __future__ import absolute_import, division, print_function, unicode_literals
from . import tree, layout, contrasts, ages
from . import bipart, genbank, nexus, newick, storage
#import nodearray, data
from . import treebase
#import gtk
#gtk.set_interactive(False)
#import db
#import contrib
try:
    from . import ltt as _ltt
    ltt = _ltt.ltt
except ImportError:
    pass

from . import chars, align, sequtil, sim, vis
## try: import vis
## except RuntimeError: pass
