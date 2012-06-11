import os, types, numpy, csv
from collections import defaultdict

class Matrix(object):
    def __init__(self):
        self.data = []
        self.col = 0
        self.fields = []

    def __getitem__(self, x):
        return self.get(x)

    def get(self, x, col=None):
        if col is None:
            col = self.col
        if x in self.data:
            return self.data[x][col]
        else:
            return None
        
    ## def add(self, data, index=0, format="csv"):
    ##     if format == "csv":
    ##         import csv
    ##         if (type(data) in types.StringTypes and
    ##             os.path.isfile(data)):
    ##             data = file(data)
    ##         r = csv.DictReader(data)
    ##         self.fields = r.fieldnames
    ##         self.data = dict()


class Rows(object):
    """
    A helper class for associating row-based data with nodes.
    Implements a dictionary-like interface that allows lookups by,
    e.g., d[node] or d['Pongo'].

    k is a function that returns the key from a row

    v is a function that returns the value from a row
    """
    def __init__(self, array=None, k=None, v=None):
        self.index = {}
        if not k: k = lambda x:x[0]
        self.k = k
        if not v: v = lambda x:x[1:] if len(x)>2 else x[1]
        self.v = v
        if array: self.index_array(array)

    def index_array(self, array):
        k = self.k; v = self.v
        for row in array:
            self.index[k(row)] = v(row)

    def __getitem__(self, item):
        "item may be a node or label"
        try: v = self.index[item]
        except KeyError:
            try: v = self.index[item.label]
            except KeyError, AttributeError:
                try: v = self.index[item.id]
                except KeyError, AttributeError:
                    try: v = self.index[item.id]
                    except KeyError, AttributeError:
                        v = None
        return v
