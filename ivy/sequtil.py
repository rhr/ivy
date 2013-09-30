from itertools import izip, imap
import numpy

def finditer(seq, substr, start=0):
    N = len(substr)
    i = seq.find(substr, start)
    while i >= 0:
        yield i
        i = seq.find(substr, i+N)

def gapidx(seq, gapchar='-'):
    """
    for a sequence with gaps, calculate site positions without gaps
    """
    a = numpy.array(seq)
    idx = numpy.arange(len(a))
    nongap = idx[a != gapchar]
    return numpy.array((numpy.arange(len(nongap)), nongap))

def find_stop_codons(seq, pos=0):
    s = seq[pos:]
    it = iter(s)
    g = imap(lambda x:"".join(x), izip(it, it, it))
    for i, x in enumerate(g):
        if x in ("TAG", "TAA", "TGA"):
            yield pos+(i*3), x
    
