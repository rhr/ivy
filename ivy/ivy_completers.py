import re
import IPython.ipapi

def node_completer(self, event):
    from tree import Node
    symbol = event.symbol
    s = event.line
    if symbol:
        s = s[:-len(symbol)]
    quote = ""
    if s and s[-1] in ["'", '"']:
        quote = s[-1]
        s = s[:-1]
    base = (re.findall(r'(\w+)\[\Z', s) or [None])[-1]

    ## print "symbol:", symbol
    ## print "line:", event.line
    ## print "s:", s
    ## print "quote:", quote
    ## print "base:", base
    ## print "obj:", self._ofind(base).get("obj")

    obj = None
    if base:
        obj = self._ofind(base).get("obj")
    if obj and isinstance(obj, Node):
        completions = ["'"]
        if quote:
            completions = sorted([ x.label for x in obj.labeled() ])
        return completions

    raise IPython.ipapi.TryNext
    
def set_node_completer():
    IPython.ipapi.get().set_hook(
        "complete_command", node_completer, re_key=r'\[*'
        )
