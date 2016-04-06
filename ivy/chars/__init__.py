import catpars, evolve
try:
    import mk
except ImportError:
    import sys
    sys.stderr.write(
        'Cannot import mk; discrete character models not available\n')
