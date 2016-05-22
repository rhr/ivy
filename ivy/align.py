from __future__ import absolute_import, division, print_function, unicode_literals

import os, types

from subprocess import Popen, PIPE
from Bio import AlignIO
from Bio.Alphabet import IUPAC
from io import StringIO
from tempfile import NamedTemporaryFile

try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]


MUSCLE = "/usr/bin/muscle"

def muscle(seqs, cmd=None):
    if not cmd: cmd = MUSCLE
    assert os.path.exists(cmd)
    p = Popen([cmd], stdin=PIPE, stdout=PIPE)
    write = p.stdin.write
    for x in seqs:
        write(">%s\n%s\n" % (x.id, x.seq))
    out = p.communicate()[0]
    aln = AlignIO.read(StringIO(out), 'fasta', alphabet=IUPAC.ambiguous_dna)
    return aln

def musclep(seqs1, seqs2, cmd="/usr/bin/muscle"):
    assert os.path.exists(cmd)
    f1 = NamedTemporaryFile(); f2 = NamedTemporaryFile()
    for s, f in ((seqs1, f1), (seqs2, f2)):
        write = f.file.write
        for x in s: write(">%s\n%s\n" % (x.id, x.seq))
    f1.file.flush(); f2.file.flush()
    cmd += " -profile -in1 %s -in2 %s" % (f1.name, f2.name)
    p = Popen(cmd.split(), stdout=PIPE)
    out = p.communicate()[0]
    aln = AlignIO.read(StringIO(out), 'fasta', alphabet=IUPAC.ambiguous_dna)
    f1.file.close(); f2.file.close()
    return aln

def read(data, format=None, name=None):

    def strip(s):
        fname = os.path.split(s)[-1]
        head, tail = os.path.splitext(fname)
        tail = tail.lower()
        if tail in (".fasta", ".nex", ".nexus"):
            return head
        else:
            return fname

    if (not format):
        if (type(data) in StringTypes) and os.path.isfile(data):
            s = data.lower()
            if s.endswith("fasta"):
                format="fasta"
            for tail in ".nex", ".nexus":
                if s.endswith(tail):
                    format="nexus"
                    break

    if (not format):
        format = "fasta"

    if type(data) in StringTypes:
        if os.path.isfile(data):
            name = strip(data)
            with open(data) as f:
                return AlignIO.read(f, format, alphabet=IUPAC.ambiguous_dna)
        else:
            f = StringIO(data)
            return AlignIO.read(f, format, alphabet=IUPAC.ambiguous_dna)

    elif (hasattr(data, "tell") and hasattr(data, "read")):
        treename = strip(getattr(data, "name", None))
        return AlignIO.read(data, format, alphabet=IUPAC.ambiguous_dna)

    raise IOError("unable to read alignment from '%s'" % data)

def write(data, f, format='fasta'):
    AlignIO.write(data, f, format)

def find(aln, substr):
    """
    generator that yields (seqnum, pos) tuples for every position of
    ``subseq`` in `aln`
    """
    from .sequtil import finditer
    N = len(substr)
    for i, rec in enumerate(aln):
        for j in finditer(rec.seq, substr):
            yield (i,j)

def find_id(aln, regexp):
    import re
    return [ (i,s) for i, s in enumerate(aln) if re.search(regexp, s.id) ]

def gapcols(aln, c='-'):
    from numpy import array
    a = array([ list(x.seq) for x in aln ])
    for i, col in enumerate(a.T):
        s = set(col==c)
        if len(s)==1 and True in s:
            yield i
