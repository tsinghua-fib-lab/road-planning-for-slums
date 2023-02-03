This is structured into three files:


1. Example.py
2. my_graph.py
3. my_graph_helpers.py



# Example.py

Example.py imports a shapefile, finds the roads necessary to create a graph
from that, and plots the graph. 


# my_graph.py

This my_graph.py file includes four classes:  `MyNode`, `MyEdge`, `MyFace`,
and `MyGraph`.


## `MyNode`

`MyNode` is a class that represents nodes. Floating point geometric inputs are
rounded to two decimal places<sup>1</sup>.  MyNodes are hashable.

1. In practice, if the map's base unit is decimal degrees, the two decimal place
rounding would be about 1.1 km at the equator, which would cause problems for
human scale neighborhoods. Reprojecting the map to meters or km, changing
significant_fig to 5, or adding the scale factor (as I've done in the other
branch) and rescaling the data appropriately would solve this.


## `MyEdge`

`MyEdge` keeps track of pairs of nodes as an edge in a graph.  Edges are
undirected. The geometric length is calculated iff called. Also has T/F
properties for being a road or barrier. Hashable.


## `MyFace`

A `MyFace` is a simple polygon, that makes up part of a planar graph.
Has area, a centroid, and a list of nodes and edges.  Not hashable.


## `MyGraph`

`MyGraph` is the bulk of the work here.  It's a wrapper around [networkx graphs](https://networkx.github.io/),
in order to make the graph explicitly spatial.  Nodes must be `MyNodes`, and are
thus located in space, in whatever the map units are and edges must by `MyEdges`.

All networkx functions are availble through `self.G`

In addition, explicitly spatial functions for self are:

1. cleaning up bad geometery (using a default threshold of 1 map unit)
2. find dual graphs
3. define roads (connected component bounding edges) and interior parcels,
as well as properties to define what nodes and edges are on roads.

Finally, the last code section can *break* the geomotery of the graph to build
in roads, rather than just defining roads as a property of some edges.  I don't
use this module, but it might be useful someday.

Several plotting and example functions are also included:

`self.plot()`  takes `networkx.draw()` keywords

`self.plot_roads()` specficially plots roads, interior parcels, and barriers.

`self.plot_weak_duals()` plots the nested dual graphs.


# my_graph_helpers.py

This file includes a bunch of helper functions for my_graph.py, with a
a variety of categories:

1. spatial geometery functions,
2. greedy search probablilty functions,
3. Ways to set up and determine the shortest paths from parcel to a road
4. importing and setting up a mygraph from a shapefile or list of existing faces
5. test functions and test graphs. 

