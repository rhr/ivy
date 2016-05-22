from __future__ import absolute_import, division, print_function, unicode_literals
import numpy

def finditer(seq, substr, start=0):
    """
    Find substrings within a sequence

    Args:
        seq (str): A sequence.
        substr (str): A subsequence to search for
        start (int): Starting index. Defaults to 0
    Yields:
        int: Starting indicies of where the substr was found in seq
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
        seq (list): Each element of the list is one character in a sequence.
        gapchar (str): The character gaps are coded as. Defaults to '-'
    Returns:
        array: An array where the first element corresponds to range(number of
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
        seq (str): A sequence
        pos (int): Starting position. Defaults to 0.
    Yields:
        tuple: The index where the stop codon starts
        and which stop codon was found.
    """
    s = seq[pos:]
    it = iter(s)
    g = map(lambda x:"".join(x), zip(it, it, it))
    for i, x in enumerate(g):
        if x in ("TAG", "TAA", "TGA"):
            yield pos+(i*3), x
