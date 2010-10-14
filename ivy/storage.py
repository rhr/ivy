import types

class Storage(dict):
    """
    A Storage object is like a dictionary except `obj.foo` can be used
    in addition to `obj['foo']`.

    From web2py/gluon/storage.py by Massimo Di Pierro (www.web2py.com)
    """
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError, k:
            raise AttributeError, k

    def __repr__(self):
        return '<Storage ' + dict.__repr__(self) + '>'

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, value):
        for (k, v) in value.items():
            self[k] = v

class CDict(dict):
    def __init__(self, cls, *args, **kwargs):
        if type(cls) == types.TypeType:
            self.cls = cls
        else:
            self.cls = cls.__class__
        dict.__init__(self, *args, **kwargs)

    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            v = self.cls()
            dict.__setitem__(self, key, v)
            return v

class SetDict(dict):
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            s = set()
            dict.__setitem__(self, key, s)
            return s

class DictDict(dict):
    def __getitem__(self, key):
        try:
            return dict.__getitem__(self, key)
        except KeyError:
            s = SetDict()
            dict.__setitem__(self, key, s)
            return s

class MaxDict(dict):
    def __setitem__(self, key, value):
        v = self.get(key)
        if value > v:
            dict.__setitem__(self, key, value)
            
            
