"""
Viewer for trees using Bokeh
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import bokeh
import ivy
import types
from bokeh.plotting import figure, output_file, show, ColumnDataSource, \
 	 reset_output
from bokeh.models import Range1d, HoverTool, BoxZoomTool, WheelZoomTool, \
     ResizeTool, ResetTool, PanTool, PreviewSaveTool
from ivy.layout import cartesian

try:
    StringTypes = types.StringTypes # Python 2
except AttributeError: # Python 3
    StringTypes = [str]
class BokehTree(object):
	def __init__(self, root, scaled = True, nodelabels = True,
				 tiplabels = True, showplot = True, hover = False):
		"""
		BokehTree class for plotting trees
		Args:
		    root (Node): A node object.
			scaled (bool): Whether or not the tree is scaled.
			  Optional, defaults to True
			nodelabels (bool): Whether or not to show node labels.
			  Optional, defaults to True.
			tiplabels (bool): Whether or not to show tip labels.
			  Optional, defaults to True.
			showplot(bool): Whether or not to display when drawtree
			  is called. Optional, defaults to True.
			hover (bool): Whether or not to use the hover tool. Optional,
			  defaults to false.
		"""
		self.root = root
		self.n2c = None
		self.xoff = 0
		self.yoff = 0
		self.coords = None # Plot coordinates for each node
		self.tools = [WheelZoomTool(),BoxZoomTool(),ResizeTool(),
					 ResetTool(),PanTool(),PreviewSaveTool()]
		self.source = None # Data source for plotting node points.
		self.nodelabels = nodelabels
		self.tiplabels = tiplabels
		self.showplot = showplot
		self.scaled = scaled
		self.hover = hover

	def set_root(self, root):
		"""
		This method sets root, leaves, and nleaves.
		Detects if tree is scaled or not.
		"""
		self.root = root
		self.leaves = root.leaves()
		self.nleaves = len(self.leaves)
		self.leaf_hsep = 1.0#/float(self.nleaves)

		for n in root.descendants():
			if n.length is None:
				self.scaled=False; break
		self.layout()

	def layout(self):
		"""
		This method calculates the coordinates of the nodes
		"""
		self.n2c = cartesian(self.root, scaled=self.scaled, yunit=1.0)
		for c in list(self.n2c.values()):
			c.x += self.xoff; c.y += self.yoff
		sv = sorted([[c.y, c.x, n] for n, c in list(self.n2c.items())])
		for i in sv:
			i[2].yval = i[0]
			i[2].xval = i[1]

		self.coords = sv

	def makehovertool(self):
		self.hovertool = HoverTool(tooltips=[("name","@desc")])


	def connectors(self):
		"""
		This method draws branches between nodes
		"""
		self.layout()
		self.labs = [xy[2].label for xy in self.coords]
		self.xLineCoords = []
		self.yLineCoords = []
		self.source = ColumnDataSource({'x':[ i[1] for i in self.coords ],
										'y':[ i[0] for i in self.coords ],
									    'desc':[ i[2].label for i in self.coords
									   ]})
		for node in self.root.postiter():
			if node.children:
				nodeParx = node.xval # Parent coordinates
				nodePary = node.yval #

				for i in node.children:
					nodeChix = i.xval # Child coordinares
					nodeChiy = i.yval #

					# Storing coordinates
					self.xLineCoords.extend([[nodeParx, nodeParx],
					                         [nodeParx, nodeChix]])
					self.yLineCoords.extend([[nodeChiy, nodePary],
					                         [nodeChiy, nodeChiy]])

		self.plot.multi_line(self.xLineCoords, self.yLineCoords)
		self.plot.circle('x', 'y', size=8, source = self.source)

	def drawlabels(self):
		"""
		This method draws labels for tips and internal nodes
		"""
		if self.nodelabels:
			# Drawing tip labels
			self.plot.text([ k.xval + 0.02*self.plot_x_range for k in
	                       self.root.leaves() ],
						   [ k.yval for k in self.root.leaves() ],
						   [ k.label for k in self.root.leaves() ],
						   text_baseline = "middle", text_font_size = "8pt")

		if self.tiplabels:
			# Drawing node labels
			self.plot.text([ k.xval + 0.02*self.plot_x_range for k in
			               self.root.clades() ],
						   [ k.yval for k in self.root.clades() ],
						   [ k.label for k in self.root.clades() ],
						   text_baseline = "middle", text_font_size = "8pt")

	def drawtree(self, showplot = True):
		"""
		This method sets the output file and creates the figure
		"""
		self.set_root(self.root)

		if self.hover:
			self.makehovertool()
			self.tools.append(self.hovertool)

		self.plot = figure(plot_width = 400, plot_height = 400,
		                   tools=self.tools)
		output_file("temp.html")

		self.plot.xgrid.grid_line_color = None # Removing grid
		self.plot.ygrid.grid_line_color = None # Removing grid

		# Drawing the branches
		self.connectors()

		if self.tiplabels == True:
			self.plot_x_range = (max([ i[1] for i in self.coords ]) -
			                     min([ i[1] for i in self.coords ]))
			self.plot_y_range = (max([ i[0] for i in self.coords ]) -
			                     min([ i[0] for i in self.coords ]))

			self.plot.x_range = Range1d( (min([ i[1] for i in self.coords ]) -
			                             (0.2*self.plot_x_range)),
										 (max([ i[1] for i in self.coords ]) +
										 (0.2*self.plot_x_range)))

		self.drawlabels()

		# Removing y axis
		self.plot.yaxis.visible = None
		# Trimming x axis
		self.plot.xaxis.bounds = (min([ i[1] for i in self.coords ]),
		                          max([ i[1] for i in self.coords ]))

		if self.showplot:
			show(self.plot) # Creating the plot

	def highlight(self, x = None, color = "red", width=3):
		"""
		Highlight selected clade(s)

		Args:
		    x: A str or list of strs or node or list of nodes.
			color (str): The color of the highlighted nodes. Optional,
			  defaults to red
			width (int): The width of the highlighted branches. Optional,
			  defaults to 3.
		"""
		if x:
			nodes = set()
			if type(x) in StringTypes:
				nodes = self.root.findall(x)
			elif isinstance(x, tree.Node):
				nodes = set(x)
			else:
				for n in x:
					if type(n) in StringTypes:
						found = self.root.findall(n)
						if found:
							nodes |= set(found)
						elif isinstance(n, tree.Node):
							nodes.add(n)

			self.highlighted = nodes
		else:
			self.highlighted = set()

		self.xHighCoords = [] # Coordinates for highlighted lines
		self.yHighCoords = [] #

		for node in self.highlighted:
			if node.parent:
				highParx = node.parent.xval # Parent coordinates
				highPary = node.parent.yval #

				highChix = node.xval # Child coordinares
				highChiy = node.yval #

				# Storing coordinates
				self.xHighCoords.extend([[highParx, highParx],
				                         [highParx, highChix]])
				self.yHighCoords.extend([[highChiy, highPary],
				                         [highChiy, highChiy]])

		self.plot.multi_line(self.xHighCoords, self.yHighCoords,
		                     color=color, line_width = width)
		self.plot.circle([ i[1] for i in self.xHighCoords ],
		                 [ i[1] for i in self.yHighCoords ],
						 color = color, size = 8)
		show(self.plot)


if __name__ == "__main__":
	r = ivy.tree.read("/home/cziegler/src/ivy/examples/primates.newick")
	f = BokehTree(r)
	f.drawtree()
