from itertools import izip, imap
import numpy

def finditer(seq, substr, start=0):
    """
    Find substrings within a sequence

    Args:
        * seq: Str. A sequence.
        * substr: Str. A subsequence to search for
        * start: Int. Starting index. Defaults to 0
    Yields:
        * Starting indicies of where the substr was found in seq
    """
    N = len(substr)
    i = seq.find(substr, start)
    while i >= 0:
        yield i
        i = seq.find(substr, i+N)

def gapidx(seq, gapchar='-'):
    """
    For a sequence with gaps, calculate site positions without gaps

    Args:
        * seq: List. Each element of the list is one character in a sequence.
        * gapchar: Str. The character gaps are coded as. Defaults to '-'
    Returns:
        * An array where the first element corresponds to range(number of
          characters that are not gaps) and the second element is the indicies
          of all characters that are not gaps.
    """
    a = numpy.array(seq)
    idx = numpy.arange(len(a))
    nongap = idx[a != gapchar]
    return numpy.array((numpy.arange(len(nongap)), nongap))

def find_stop_codons(seq, pos=0):
    """
    Find stop codons within sequence (in reading frame)

    Args:
        * seq: Str. A sequence
        * pos: Int. Starting position. Defaults to 0.
    Yields:
        * Tuple containing the index where the stop codon starts
          and which stop codon was found.
    """
    s = seq[pos:]
    it = iter(s)
    g = imap(lambda x:"".join(x), izip(it, it, it))
    for i, x in enumerate(g):
        if x in ("TAG", "TAA", "TGA"):
            yield pos+(i*3), x
