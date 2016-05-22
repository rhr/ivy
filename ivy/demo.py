from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from IPython.lib.demo import IPythonDemo, ClearIPDemo
## from IPython.genutils import Term
from IPython.utils.io import stdout as cout
from time import sleep
from random import random

class Demo(ClearIPDemo):
    def show(self,index=None):
        """Show a single block on screen"""

        index = self._get_index(index)
        if index is None:
            return

        if self.speed:
            r = 1.0/self.speed
            for c in self.src_blocks_colored[index]:
                cout.write(c)
                sleep(random()*0.1*r)
                sys.stdout.flush()
        else:
            print(self.src_blocks_colored[index], file=cout)
            sys.stdout.flush()

        s = "Hit <Enter>, then type %next to proceed or %back to go back one step."
        print(s, file=cout)
        cout.flush()
    

def demo(src, speed=1):
    __demo__ = Demo(src)
    __demo__.speed = speed
    def n(*args): __demo__()
    def b(*args):
        __demo__.back(2)
        __demo__()
    ## ip = IPython.ipapi.get()
    ip = get_ipython()
    ip.define_magic('n', n)
    ip.define_magic('next', n)
    ip.define_magic('b', b)
    ip.define_magic('back', b)
    __demo__()
