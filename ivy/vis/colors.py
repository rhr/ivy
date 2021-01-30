## from matplotlib import cm as mpl_colormap
## from matplotlib import colors as mpl_colors
import pandas as pd
from itertools import cycle
from io import StringIO

tango_colors = {
    'Aluminium1': (0.933, 0.933, 0.925, 1),
    'Aluminium2': (0.827, 0.843, 0.812, 1),
    'Aluminium3': (0.729, 0.741, 0.714, 1),
    'Aluminium4': (0.533, 0.541, 0.522, 1),
    'Aluminium5': (0.333, 0.341, 0.325, 1),
    'Aluminium6': (0.180, 0.204, 0.212, 1),
    'Butter1': (0.988, 0.914, 0.310, 1),
    'Butter2': (0.929, 0.831, 0.000, 1),
    'Butter3': (0.769, 0.627, 0.000, 1),
    'Chameleon1': (0.541, 0.886, 0.204, 1),
    'Chameleon2': (0.451, 0.824, 0.086, 1),
    'Chameleon3': (0.306, 0.604, 0.024, 1),
    'Chocolate1': (0.914, 0.725, 0.431, 1),
    'Chocolate2': (0.757, 0.490, 0.067, 1),
    'Chocolate3': (0.561, 0.349, 0.008, 1),
    'Orange1': (0.988, 0.686, 0.243, 1),
    'Orange2': (0.961, 0.475, 0.000, 1),
    'Orange3': (0.808, 0.361, 0.000, 1),
    'Plum1': (0.678, 0.498, 0.659, 1),
    'Plum2': (0.459, 0.314, 0.482, 1),
    'Plum3': (0.361, 0.208, 0.400, 1),
    'ScarletRed1': (0.937, 0.161, 0.161, 1),
    'ScarletRed2': (0.800, 0.000, 0.000, 1),
    'ScarletRed3': (0.643, 0.000, 0.000, 1),
    'SkyBlue1': (0.447, 0.624, 0.812, 1),
    'SkyBlue2': (0.204, 0.396, 0.643, 1),
    'SkyBlue3': (0.125, 0.290, 0.529, 1),
    }

def tango():
    c = cycle(map(tango_colors.get,
                  ("ScarletRed3", "SkyBlue3", "Chameleon3", "Plum3",
                   "Orange3", "Butter3", "Chocolate3", "Aluminium6",
                   "ScarletRed1", "SkyBlue1", "Chameleon1", "Plum1",
                   "Orange1", "Butter1", "Chocolate1", "Aluminium4")))
    return c

table = pd.read_csv(StringIO("""
name,light,medium,dark
blue,#729fcf,#3465a4,#204a87
orange,#fcaf3e,#f57900,#ce5c00
green,#8ae234,#73d216,#4e9a06
purple,#ad7fa8,#75507b,#5c3566
brown,#e9b96e,#c17d11,#8f5902
red,#ef2929,#cc0000,#a40000
yellow,#fce94f,#edd400,#c4a000
darkgray,#888a85,#555753,#2e3436
lightgray,#eeeeec,#d3d7cf,#babdb6
"""),index_col='name')

light = table.light
medium = table.medium
dark = table.dark
