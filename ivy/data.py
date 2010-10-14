import os, types

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

    
