import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import itertools
import math
import warnings
import json
import pretreatment.my_graph_helpers as mgh
from pretreatment.lazy_property import lazy_property

import random,time
#import plotly.plotly as py
#from plotly.graph_objs import *
"""
This my_graph.py file includes three classes:  MyNode, MyEdge, MyFace,
and MyGraph.


MyNode

MyNode is a class that represents nodes. Floating point geometric inputs are
rounded to two decimal places*.  MyNodes are hashable.

*In practice, if the map's base unit is decimal degrees, the two decimal place
rounding would be about 1.1 km at the equator, which could be problematic.
reprojecting the map to meters or km would solve this problem, or changing
significant_fig to 5 would solve this.


MyEdge

MyEdge keeps track of pairs of nodes as an edge in a graph.  Edges are
undirected. The geometric length is calculated if called. Also has T/F
properties for being a road or barrier. Hashable.


MyFace

A myface is essentially a simple polygon, that makes up part of a planar graph.
Has area, a centroid, and a list of nodes and edges.  Not hashable.

MyGraph

MyGraph is the bulk of the work here.  It's a wrapper around networkx graphs,
to be explicitly spatial.  Nodes must by MyNodes, and so located in space,
and edges must by MyEdges.

All networkx functions are availble through myG.G

In addition, explicitly spatial functions for myG are:
1) cleaning up bad geometery
2) find dual graphs
3) define roads (connected component bounding edges) and interior parcels,
as well as properties to define what nodes and edges are on roads.

Finally, the last code section can "break" the geomotery of the graph to build
in roads, rather than just defining roads as a property of some edges.  I don't
use this module, but it might be useful someday.

Several plotting and example functions are also included:

myG.plot()  takes normal networkx.draw() keywords

myG.plot_roads specficially plots roads, interior parcels, and barriers.

myG.plot__weak_duals plots the nexted dual graphs.

"""


class MyNode(object):
    """ rounds float nodes to (2!) decimal places, defines equality """

    def __init__(self, locarray, name=None):
        significant_figs = 2
        if len(locarray) != 2:
            print("error")
        x = locarray[0]
        y = locarray[1]
        self.x = np.round(float(x), significant_figs)
        self.y = np.round(float(y), significant_figs)
        self.loc = (self.x, self.y)
        self.road = False
        self.interior = False
        self.barrier = False
        self.name = name

    def __repr__(self):
        if self.name:
            return self.name
        else:
            return "(%.2f,%.2f)" % (self.x, self.y)

    def __eq__(self, other):
        if hasattr(other, 'loc'):
            return self.loc == other.loc
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.loc < other.loc

    def __hash__(self):
        return hash(self.loc)


class MyEdge(object):
    """ keeps the properties of the edges in a parcel."""

    def __init__(self, nodes):
        self.nodes = tuple(nodes)
        self.interior = False
        self.road = False
        self.barrier = False

    @lazy_property
    def length(self):
        return mgh.distance(self.nodes[0], self.nodes[1])

    @lazy_property
    def rads(self):
        return math.atan((self.nodes[0].y - self.nodes[1].y) /
                         (self.nodes[0].x - self.nodes[1].x))

    def __repr__(self):
        return "MyEdge with nodes {} {}".format(self.nodes[0], self.nodes[1])

    def __eq__(self, other):
        return ((self.nodes[0] == other.nodes[0]
                 and self.nodes[1] == other.nodes[1])
                or (self.nodes[0] == other.nodes[1]
                    and self.nodes[1] == other.nodes[0]))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.nodes)

    def geoJSON(self, rezero, rescale):
        return {
            "type": "Feature",
            "geometry": {
                "type":
                "LineString",
                "coordinates": [
                    list([(n.x + rezero[0]) / rescale[0],
                          (n.y + rezero[1]) / rescale[1]]) for n in self.nodes
                ]
            },
            "properties": {
                "road": str(self.road).lower(),
                "interior": str(self.interior).lower(),
                "barrier": str(self.barrier).lower()
            }
        }


class MyFace(object):
    """class defines a face (with name and list of edges & nodes)
       from a list of edges in the face"""

    def __init__(self, list_of_edges):
        # make a list of all the nodes in the face

        isMyEdge = False
        if len(list_of_edges) > 0:
            isMyEdge = type(list_of_edges[0]) != tuple

        if isMyEdge:
            node_set = set(n for edge in list_of_edges for n in edge.nodes)
        else:
            node_set = set(n for edge in list_of_edges for n in edge)

        self.nodes = sorted(list(node_set))
        alpha_nodes = map(str, self.nodes)
        self.name = ".".join(alpha_nodes)
        self.paths = None
        self.on_road = False
        self.even_nodes = {}
        self.odd_node = {}

        # the position of the face is the centroid of the nodes that
        # compose the face

        if isMyEdge:
            self.edges = set(list_of_edges)
            self.ordered_edges = list_of_edges
        else:
            self.edges = set(MyEdge(e) for e in list_of_edges)
            self.ordered_edges = [MyEdge(e) for e in list_of_edges]

    @lazy_property
    def area(self):
        return 0.5 * abs(
            sum(e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y
                for e in self.ordered_edges))

    @lazy_property
    def centroid(self):
        """finds the centroid of a MyFace, based on the shoelace method
        e.g. http://en.wikipedia.org/wiki/Shoelace_formula and
        http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
        The method relies on properly ordered edges. """

        a = 0.5 * (sum(
            e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y
            for e in self.ordered_edges))
        if abs(a) < 0.01:
            cx = np.mean([n.x for n in self.nodes])
            cy = np.mean([n.y for n in self.nodes])
        else:
            cx = (1 / (6 * a)) * sum(
                [(e.nodes[0].x + e.nodes[1].x) *
                 (e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y)
                 for e in self.ordered_edges])
            cy = (1 / (6 * a)) * sum(
                [(e.nodes[0].y + e.nodes[1].y) *
                 (e.nodes[0].x * e.nodes[1].y - e.nodes[1].x * e.nodes[0].y)
                 for e in self.ordered_edges])

        return MyNode((cx, cy))

    def __len__(self):
        return len(self.edges)

    def __repr__(self):
        return "Face with centroid at (%.2f,%.2f)" % (self.centroid.x,
                                                      self.centroid.y)


class MyGraph(object):

    def __init__(self, G=None, name="S0"):
        """ MyGraph is a regular networkx graph where nodes are stored
        as MyNodes and edges have the attribute myedge = MyEdge.

        The extra function weak_dual() finds the weak dual
        (http://en.wikipedia.org/wiki/Dual_graph#Weak_dual) of the
        graph based on the locations of each node.  Each node in the
        dual graph corresponds to a face in G, the position of each
        node in the dual is caluclated as the mean of the nodes
        composing the corresponding face in G."""

        self.name = name
        self.cleaned = False
        self.roads_update = True
        self.rezero_vector = np.array([0, 0])
        self.rescale_vector = np.array([1, 1])
        self.td_dict = {}

        if G is None:
            self.G = nx.Graph()
        else:
            self.G = G

    def __repr__(self):
        return "Graph (%s) with %d nodes" % (self.name,
                                             self.G.number_of_nodes())

    def add_node(self, n):
        self.G.add_node(n)

    def add_edge(self, e, weight=None):
        assert isinstance(e, MyEdge)
        if weight is None:
            w = e.length
        else:
            w = weight
        self.G.add_edge(e.nodes[0], e.nodes[1], myedge=e, weight=w)

    def _fake_edge(self, centroid, mynode):
        newedge = MyEdge((centroid, mynode))
        newedge.length = 0
        self.add_edge(newedge)

    def add_fake_edges(self, p):
        [self._fake_edge(p.centroid, n) for n in p.nodes]

    def location_dict(self):
        return dict((n, n.loc) for n in self.G.nodes())

    def connected_components(self):
        return [
            MyGraph(g, self.name) for i, g in enumerate([
                self.G.subgraph(c).copy()
                for c in nx.connected_components(self.G)
            ])
        ]

    def myedges(self):
        return [self.G[e[0]][e[1]]["myedge"] for e in self.G.edges()]

    def myweight(self):
        return [self.G[e[0]][e[1]]["weight"] for e in self.G.edges()]

    def remove_myedges_from(self, myedges):
        myedge_tups = [(e.nodes[0], e.nodes[1]) for e in myedges]
        self.G.remove_edges_from(myedge_tups)

    def copy(self):
        """  Relies fundamentally on nx.copy function.  This creates a copy of
        the nx graph, where the nodes and edges retain their properties.
        MyGraph properties have to be recalculated, because copy needs to make
        entirely new faces and face attributes.
        """

        nx_copy = self.G.copy()
        copy = MyGraph(nx_copy)
        copy.name = self.name
        copy.rezero_vector = self.rezero_vector
        copy.rescale_vector = self.rescale_vector
        copy.td_dict = self.td_dict
        # outerface is a side effect of the creation of inner_facelist
        # so we operate on that in order to not CALL inner_facelist for every
        # copy.
        if hasattr(self, 'outerface'):
            copy.inner_facelist

        # order matters.  road nodes before interior parcels
        if hasattr(self, 'road_nodes'):
            copy.road_nodes = [n for n in copy.G.nodes() if n.road]

        if hasattr(self, 'road_edges'):
            copy.road_edges = [e for e in copy.myedges() if e.road]

        if hasattr(self, 'interior_parcels'):
            copy.define_interior_parcels()

        return copy

    @lazy_property
    def inner_facelist(self):
        inner_facelist = self.__trace_faces()
        # print "inner_facelist called for graph {}".format(self)
        return inner_facelist

    def myedges_geoJSON(self):
        return json.dumps({
            "type":
            "FeatureCollection",
            "features": [
                e.geoJSON(self.rezero_vector, self.rescale_vector)
                for e in self.myedges()
            ]
        })

############################
# GEOMETRY CLEAN UP FUNCTIONS
############################

    def __merge_line_nodes(self, err):
        for n in list(self.G.nodes()):

            def angle(b):
                dx = b.x - n.x
                dy = b.y - n.y
                return np.arctan2(dx, dy)

            def is_line(nlist, err):
                err = err
                an1 = abs(angle(nlist[0]) - angle(nlist[1]))
                an2 = abs(angle(nlist[0]) + angle(nlist[1]))
                if abs(an1 - np.pi) < err or an1 < err:
                    return True
                else:
                    return False

            ng = list(self.G.neighbors(n))

            if len(ng) == 2:
                w1 = self.G[ng[0]][n]['weight']
                w2 = self.G[ng[1]][n]['weight']
                self.G.remove_node(n)
                self.G.add_edge(ng[0],
                                ng[1],
                                myedge=MyEdge((ng[0], ng[1])),
                                weight=w1 + w2)

            elif len(ng) < 2:
                self.G.remove_node(n)




    def __combine_near_nodes(self, threshold):
        """takes a connected component MyGraph, finds all nodes that are
        within a threshold distance of each other, drops one and keeps the
        other, and reconnects all the nodes that were connected to the first
        node to the second node.  """

        nlist = self.G.nodes()
        for i, j in itertools.combinations(nlist, 2):
            if j in self.G and i in self.G:
                if mgh.distance_squared(i, j) < threshold**2:
                    drop = j
                    keep = i
                    neighbors_drop = self.G.neighbors(drop)
                    neighbors_keep = self.G.neighbors(keep)
                    edges_to_add = (set(neighbors_drop) - set([keep]) -
                                    set(neighbors_keep))
                    self.G.remove_node(drop)
                    for k in edges_to_add:
                        newedge = MyEdge((keep, k))
                        self.add_edge(newedge)

    def __find_bad_edges(self, threshold):
        """ finds nodes that are within the threshold distance of an edge
        that does not contain it. Does not pair node V to edge UV.

        Returns a dict with edges as keys, and the node that is too close as
        the value.  This might cause trouble, if there are nodes that just
        should be collapsed together, rather than the edge being split in
        order to get that node connected. """

        node_list = self.G.nodes()
        edge_tup_list = self.G.edges(data=True)
        edge_list = [e[2]['myedge'] for e in edge_tup_list]
        bad_edge_dict = {}
        for i in node_list:
            for e in edge_list:
                if i != e.nodes[0] and i != e.nodes[1]:
                    # print "{} IS NOT on {}".format(i,e)
                    node_dist_sq = mgh.sq_distance_point_to_segment(i, e)
                    if node_dist_sq < threshold**2:
                        # print "{} is too close to {}".format(i, e)
                        if e in bad_edge_dict:
                            bad_edge_dict[e].append(i)
                        else:
                            bad_edge_dict[e] = list([i])
        self.bad_edge_dict = bad_edge_dict
        return

    def __remove_bad_edges(self, bad_edge_dict):
        """ From the dict of bad edges:  call edge (dict key) UV and the node
        (dict value) J. Then, drop edge UV and ensure that there is an edge
        UJ and JV.
        """

        dropped_edges = 0
        for edge, node_list in bad_edge_dict.items():
            # print "dropping edge {}".format((edge.nodes[0],edge.nodes[1]))
            self.G.remove_edge(edge.nodes[0], edge.nodes[1])
            dropped_edges = dropped_edges + 1
            if len(node_list) == 1:
                for j in [0, 1]:
                    if not self.G.has_edge(node_list[0], edge.nodes[j]):
                        self.add_edge(MyEdge((node_list[0], edge.nodes[j])))
            else:
                node_list.sort(
                    key=lambda node: mgh.distance(node, edge.nodes[0]))
                if not self.G.has_edge(node_list[0], edge.nodes[0]):
                    self.add_edge(MyEdge((node_list[0], edge.nodes[0])))
                for i in range(1, len(node_list)):
                    if not self.G.has_edge(node_list[i], node_list[i - 1]):
                        self.add_edge(MyEdge((node_list[i], node_list[i - 1])))
                if not self.G.has_edge(node_list[-1], edge.nodes[1]):
                    self.add_edge(MyEdge((node_list[-1], edge.nodes[1])))

        return dropped_edges

    def clean_up_geometry(self, threshold, err=0.2, byblock=True):
        """ function cleans up geometry, and returns a _copy_  of the graph,
        cleaned up nicely. Does not change original graph. connected considers
        graph by connected components only for clean up.
        """

        Gs = []

        i = self.copy()
        i.G.remove_edges_from(nx.selfloop_edges(i.G))
        i.__combine_near_nodes(threshold)
        # i.__find_bad_edges(threshold)
        # i.__remove_bad_edges(i.bad_edge_dict)
        # i.__merge_line_nodes(err)
        # i.__merge_line_nodes(err)
        Gs.append(i.G)

        nxG = nx.compose_all(Gs)
        newG = MyGraph(nxG, name=self.name)
        newG.cleaned = True

        return newG

    def clean_up_geometry_single_CC(self, threshold):
        """ function cleans up geometry, and returns a _copy_  of the graph,
        cleaned up nicely. Does not change original graph.
        """

        Gs = self.copy()

        Gs.G.remove_edges_from(nx.selfloop_edges(Gs))
        Gs.__combine_near_nodes(threshold)
        Gs.__find_bad_edges(threshold)
        Gs.__remove_bad_edges(Gs.bad_edge_dict)

        Gs.name = self.name

        Gs.cleaned = True

        return Gs

    def save_graph(self, file_path):
        print(file_path)
        nx.write_shp(self.G, file_path)
##########################################
#    WEAK DUAL CALCULATION FUNCTIONS
########################################

    def get_embedding(self):
        emb = {}
        for i in self.G.nodes():
            neighbors = self.G.neighbors(i)

            def angle(b):
                dx = b.x - i.x
                dy = b.y - i.y
                return np.arctan2(dx, dy)

            reorder_neighbors = sorted(neighbors, key=angle)
            emb[i] = reorder_neighbors
        return emb

    def __trace_faces(self):
        """Algorithm from SAGE"""
        if len(self.G.nodes()) < 2:
            inner_facelist = []
            return []

        # grab the embedding
        comb_emb = self.get_embedding()

        # Establish set of possible edges
        edgeset = set()
        for edge in self.G.edges():
            edgeset = edgeset.union(
                set([(edge[0], edge[1]), (edge[1], edge[0])]))

        # Storage for face paths
        faces = []

        # Trace faces
        face = [edgeset.pop()]
        while (len(edgeset) > 0):
            neighbors = comb_emb[face[-1][-1]]
            next_node = neighbors[(neighbors.index(face[-1][-2]) + 1) %
                                  (len(neighbors))]
            edge_tup = (face[-1][-1], next_node)
            if edge_tup == face[0]:
                faces.append(face)
                face = [edgeset.pop()]
            else:
                face.append(edge_tup)
                edgeset.remove(edge_tup)

        if len(face) > 0:
            faces.append(face)

        # remove the outer "sphere" face
        facelist = sorted(faces, key=len)
        self.outerface = MyFace(facelist[-1])
        self.outerface.edges = [
            self.G[e[1]][e[0]]["myedge"] for e in facelist[-1]
        ]
        inner_facelist = []
        for face in facelist[:-1]:
            iface = MyFace(face)
            iface.edges = [self.G[e[1]][e[0]]["myedge"] for e in face]
            inner_facelist.append(iface)
            iface.down1_node = iface.centroid
        return inner_facelist

    def weak_dual(self):
        """This function will create a networkx graph of the weak dual
        of a planar graph G with locations for each node.Each node in
        the dual graph corresponds to a face in G. The position of each
        node in the dual is caluclated as the mean of the nodes composing
        the corresponding face in G."""

        try:
            assert len(list(self.G.subgraph(g) for g in nx.connected_components(self.G))) <= 1
        except AssertionError:
            raise RuntimeError("weak_dual() can only be called on" +
                               " graphs which are fully connected.")

        # name the dual
        if len(self.name) == 0:
            dual_name = ""
        else:
            lname = list(self.name)
            nums = []
            while True:
                try:
                    nums.append(int(lname[-1]))
                except ValueError:
                    break
                else:
                    lname.pop()

            if len(nums) > 0:
                my_num = int(''.join(map(str, nums)))
            else:
                my_num = -1
            my_str = ''.join(lname)
            dual_name = my_str + str(my_num + 1)

        # check for empty graph
        if self.G.number_of_nodes() < 2:
            return MyGraph(name=dual_name)

        # get a list of all faces
        # self.trace_faces()

        # make a new graph, with faces from G as nodes and edges
        # if the faces share an edge
        dual = MyGraph(name=dual_name)
        if len(self.inner_facelist) == 1:
            face = self.inner_facelist[0]
            dual.add_node(face.centroid)
        else:
            combos = list(itertools.combinations(self.inner_facelist, 2))
            for c in combos:
                c0 = [e for e in c[0].edges if not e.road]
                c1 = [e for e in c[1].edges if not e.road]
                if len(set(c0).intersection(c1)) > 0:
                    dual.add_edge(MyEdge((c[0].centroid, c[1].centroid)))
        return dual

    def S1_nodes(self):
        """Gets the odd_node dict started for depth 1 (all parcels have a
        centroid) """
        for f in self.inner_facelist:
            f.odd_node[1] = f.centroid

    def formClass(self, duals, depth, result):
        """ function finds the groups of parcels that are represented in the
        dual graph with depth "depth+1".  The depth value provided must be even
        and less than the max depth of duals for the graph.

        need to figure out why I can return a result with depth d+1 with an
        empty list.

        """

        dm1 = depth - 1

        is_odd = bool(depth % 2)

        try:
            assert not is_odd
        except AssertionError:
            raise RuntimeError("depth ({}) should be even".format(depth))

        # flist is the list of parcels in self which are represented in the
        # dual of depth depth-1 (dm1)
        flist = [
            f for f in self.inner_facelist
            if (dm1 in f.odd_node and f.odd_node[dm1])
        ]

        dual1 = duals[dm1]
        dual2 = duals[depth]

        # flat list of faces in duals 1 and 2 for potentially many disconnected
        # dual graphs.
        dual1_faces = [f for G in dual1 for f in G.inner_facelist]
        dual2_faces = [f for G in dual2 for f in G.inner_facelist]

        # creates an association between the faces in self and the centroids
        # of faces in dual1, for faces in dual1 that overlap a face (face0) in
        # self.
        for face0 in flist:
            down2_nodes = [
                f.centroid for f in dual1_faces
                if face0.odd_node[depth - 1] in f.nodes
            ]
            face0.even_nodes[depth] = set(down2_nodes)
#            down2_nodes = []
#            for face1 in dual1_faces:
#                if face0.odd_node[depth-1] in face1.nodes:
#                    down2_nodes.append(face1.centroid)
#                    face0.even_nodes[depth] = set(down2_nodes)

# if the down2 faces for face0 make up a face in the dual2 graph, then
# the centroid of that face in the dual2 graph represents face0 in the
# dual graph with depth depth+1
        for face0 in flist:
            if depth in face0.even_nodes:
                for face2 in dual2_faces:
                    if set(face0.even_nodes[depth]) == set(face2.nodes):
                        face0.odd_node[depth + 1] = face2.centroid

        # return the results as a dict for depth depth+1, also stored as a
        # a property of each face.
        result[depth + 1] = [
            f for f in self.inner_facelist
            if depth + 1 in f.odd_node and f.odd_node[depth + 1]
        ]

        depth = depth + 2
        return duals, depth, result

    def stacked_duals(self, maxdepth=15):
        """to protect myself from an infinite loop, max depth defaults to 15"""

        def level_up(Slist):
            Sns = [g.weak_dual().connected_components() for g in Slist]
            Sn = [cc for duals in Sns for cc in duals]
            return Sn

        stacks = []
        stacks.append([self])
        while len(stacks) < maxdepth:
            slist = level_up(stacks[-1])
            if len(slist) == 0:
                break
            stacks.append(slist)

        for G in stacks:
            for g in G:
                try:
                    g.inner_facelist
                except AttributeError:
                    g.__trace_faces()
                    print("tracing faces needed")

        return stacks

#############################################
#  DEFINING ROADS AND INTERIOR PARCELS
#############################################

    def define_roads(self):
        """ finds which edges and nodes in the connected component are on
        the roads, and updates thier properties (node.road, edge.road) """

        road_nodes = []
        road_edges = []
        # check for empty graph
        if self.G.number_of_nodes() < 2:
            return []

        # self.trace_faces()
        self.inner_facelist
        of = self.outerface
        for e in of.edges:
            e.road = True
            road_edges.append(e)
        for n in of.nodes:
            n.road = True
            road_nodes.append(n)
        self.roads_update = True
        self.road_nodes = road_nodes
        self.road_edges = road_edges
        print('nodes:',len(self.G.nodes),'edges:',len(self.G.edges))
        

        # print "define roads called"

    def define_interior_parcels(self):
        """defines what parcels are on the interior based on
           whether their nodes are on roads.  Relies on self.inner_facelist
           and self.road_nodes being updated. Writes to self.interior_parcels
           and self.interior_nodes
           """

        if self.G.number_of_nodes() < 2:
            return []

        interior_parcels = []

        for n in self.G.nodes():
            mgh.is_roadnode(n, self)

        self.road_nodes = [n for n in self.G.nodes() if n.road]

        # rewrites all edge properties as not being interior.This needs
        # to happen BEFORE we define the edge properties for parcels
        # that are interior, in order to give that priority.
        for e in self.myedges():
            e.interior = False

        for f in self.inner_facelist:
            if len(set(f.nodes).intersection(set(self.road_nodes))) == 0:
                f.on_road = False
                interior_parcels.append(f)
            else:
                f.on_road = True
                for n in f.nodes:
                    n.interior = False

        for p in interior_parcels:
            for e in p.edges:
                e.interior = True

        for n in self.G.nodes():
            mgh.is_interiornode(n, self)

        self.interior_parcels = interior_parcels
        self.interior_nodes = [n for n in self.G.nodes() if n.interior]

        self.td_dict_init()
        # print "define interior parcels called"

    def update_node_properties(self):
        for n in self.G.nodes():
            mgh.is_roadnode(n, self)
            mgh.is_interiornode(n, self)
            mgh.is_barriernode(n, self)

    def find_interior_edges(self):
        """ finds and returns the pairs of nodes (not the myEdge) for all edges that
        are not on roads."""

        interior_etup = []

        for etup in self.G.edges():
            if not self.G[etup[0]][etup[1]]["myedge"].road:
                interior_etup.append(etup)

        return interior_etup

    def road_update(self, edge):
        self.G[edge.nodes[0]][edge.nodes[1]]['road'] = self.G[edge.nodes[0]][
            edge.nodes[1]]['weight']

    def add_road_segment(self, edge):
        """ Updates properties of graph to make edge a road. """
        edge.road = True

        if hasattr(self, 'road_edges'):
            self.road_edges.append(edge)
        else:
            self.road_edges = [edge]

        if hasattr(self, 'road_nodes'):
            rn = self.road_nodes
        else:
            rn = []

        for n in edge.nodes:
            n.road = True
            rn.append(n)

        self.roads_update = False
        self.road_nodes = rn
        self.define_interior_parcels()
        # self.road_update(edge)
        # t1 = time.perf_counter()
        # self.interior_parcels_update(edge)
        # self.td_dict_update(edge)
        # self.interior_parcels_update(edge)
        # print(time.perf_counter() -t1)

    def remove_road_segment(self, edge):
        """ Updates properties of graph to remove a road. """
        assert isinstance(edge, MyEdge)
        edge.road = False
        for n in edge.nodes:
            onroad = False
            for neighbor in self.G[n]:
                neighboredge = self.G[n][neighbor]['myedge']
                if neighboredge.road:
                    onroad = True

            n.road = onroad
            if not n.road:
                if n in self.road_nodes:
                    self.road_nodes.remove(n)

        self.define_interior_parcels()
        return

    def road_length(self):
        """finds total length of roads in self """
        eroad = [e for e in self.myedges() if e.road]
        length = sum([e.length for e in eroad])
        return length

    def road_cost(self):
        """finds total length of roads in self """
        eroad = [e for e in self.myedges() if e.road]
        length = sum([self.G[e[0]][e[1]]['weight'] for e in eroad])
        return length

    def travel_distance(self):
        return self._td_reward / self._td_reward_count

    def td_dict_init(self):
        roadG = MyGraph()
        for e in self.road_edges:
            roadG.add_edge(e)
        roadf = [f for f in self.inner_facelist if f.on_road]
        for f in roadf:
            roadG.add_fake_edges(f)

        td_dict = dict(nx.shortest_path_length(roadG.G, weight="weight"))
        self.td_dict = {}
        for i in roadG.G.nodes():
            self.td_dict[i] = {}
            for j in roadf:
                self.td_dict[i][j] = td_dict[i][j.centroid]

    def td_dict_update(self, edge):
        n1 = edge.nodes[0]
        n2 = edge.nodes[1]
        road_length = self.G[n1][n2]['weight']
        self._td_reward = 0
        self._td_reward_count = 0
        print (len(self.td_dict[n1]) , len(self.td_dict[n2]))
        if n1 not in self.td_dict:
            if n2 not in self.td_dict:
                raise ValueError('road is not connected to original graph')
            else:
                self.td_dict[n1] = {}
                for f in self.inner_facelist:
                    if n1 in set(f.nodes):
                        if f in self.td_dict[n2]:
                            self.td_dict[n2][f] = min(self.td_dict[n2][f],
                                                      road_length)
                        else:
                            self.td_dict[n2][f] = self.G[n1][n2]['weight']

                        for ff in self.td_dict[n2]:
                            self.td_dict[n1][
                                ff] = self.td_dict[n2][ff] + road_length
                self._td_reward = 1 / len(self.inner_facelist)

        elif n2 not in self.td_dict:
            if n1 not in self.td_dict:
                raise ValueError('road is not connected to original graph')
            else:
                self.td_dict[n2] = {}
                for f in self.td_dict[n1]:
                    if n1 in set(f.nodes):
                        if f in self.td_dict[n1]:
                            self.td_dict[n1][f] = min(self.td_dict[n1][f],
                                                      road_length)
                        else:
                            self.td_dict[n1][f] = self.G[n1][n2]['weight']

                        for ff in self.td_dict[n1]:
                            self.td_dict[n2][
                                ff] = self.td_dict[n1][ff] + road_length
                self._td_reward = 1 / len(self.inner_facelist)

        else:
            for f in self.td_dict[n1]:
                tmp = self.td_dict[n1][f]
                if tmp != 0:
                    self.td_dict[n1][f] = min(
                        self.td_dict[n1][f], self.td_dict[n2][f] + road_length)
                    self._td_reward += (tmp - self.td_dict[n1][f]) / tmp
                    self._td_reward_count += 1
            for f in self.td_dict[n2]:
                tmp = self.td_dict[n2][f]
                if tmp != 0:
                    self.td_dict[n2][f] = min(
                        self.td_dict[n2][f], self.td_dict[n1][f] + road_length)
                    self._td_reward += (tmp - self.td_dict[n1][f]) / tmp
                    self._td_reward_count += 1
        print (len(self.td_dict[n1]) , len(self.td_dict[n2]))

        
    def interior_parcels_update(self, edge):
        for f in self.interior_parcels:
            if set(edge.nodes).intersection(set(f.nodes)):
                self.interior_parcels.remove(f)
                print('del')

#############################################
#   GEOMETRY AROUND BUILDING A GIVEN ROAD SEGMENT - c/(sh?)ould be deleted.
#############################################

    def __find_nodes_curbs(self, edge):
        """ finds curbs and nodes for a given edge that ends on a road.
        """

        if edge.nodes[0].road == edge.nodes[1].road:
            raise Exception("{} does not end on a curb".format(edge))

        [b] = [n for n in edge.nodes if n.road]
        [a] = [n for n in edge.nodes if not n.road]

        b_neighbors = self.G.neighbors(b)

        curb_nodes = [n for n in b_neighbors if self.G[b][n]["myedge"].road]

        if len(curb_nodes) != 2:
            raise Exception("Trouble!  " +
                            "Something is weird about the road geometery.")

        [c1, c2] = curb_nodes

        return a, b, c1, c2

    def __find_d_connections(self, a, b, c1, c2, d1, d2):
        """ nodes d1 and d2 are added to graph, and figures
            out how a, c1 and c2 are connected  """

        for n in [d1, d2]:
            self.add_edge(MyEdge((n, b)))

        emb = self.get_embedding()

        Bfilter = [n for n in emb[b] if n in [a, c1, c2, d1, d2]]

        if len(Bfilter) != 5:
            raise Exception(
                "Bfilter is not set up correctly. \n {}".format(Bfilter))

        Aindex = Bfilter.index(a)

        while Aindex != 2:
            mgh.myRoll(Bfilter)
            Aindex = Bfilter.index(a)

        newedges = []
        newedges.append(MyEdge((a, d1)))
        newedges.append(MyEdge((a, d2)))
        newedges.append(MyEdge((Bfilter[0], Bfilter[1])))
        newedges.append(MyEdge((Bfilter[3], Bfilter[4])))

        return newedges

    def __find_e_connections(self, a, b, c1, c2, d1, d2):
        """ of nodes connected to b that are not existing curbs (c1 and c2)
        and a, the endpoint of the new road segment, figures out how to
        connect each one to d1 or d2 (the new curbs).   """

        emb = self.get_embedding()
        Bfilter = [n for n in emb[b] if n not in [d1, d2]]

        # if c1 and c2 are the first two elements, roll so they are the
        # first and last
        if ((Bfilter[0] == c1 or Bfilter[0] == c2)
                and (Bfilter[1] == c1 or Bfilter[1] == c2)):
            mgh.myRoll(Bfilter)

        # roll until c1 or c2 is first.  the other should then be the
        # last element in the list.
        while Bfilter[0] != c1 and Bfilter[0] != c2:
            mgh.myRoll(Bfilter)

        # check that after rolling, c1 and c2 are first and last elements
        if Bfilter[-1] != c1 and Bfilter[-1] != c2:
            raise Exception("There is an edge in my road." +
                            "Something is wrong with the geometry.")

        # d1 connected to c1 or c2?
        if c1 in self.G[d1]:
            c1_to_d1 = True
        else:
            c1_to_d1 = False

        if Bfilter[0] == c1:
            if c1_to_d1:
                dorder = [d1, d2]
            else:
                dorder = [d2, d1]
        elif Bfilter[0] == c2:
            if c1_to_d1:
                dorder = [d2, d1]
            else:
                dorder = [d1, d2]
        else:
            raise Exception("Bfilter is set up wrong")

        Aindex = Bfilter.index(a)

        newedges1 = [MyEdge((dorder[0], n)) for n in Bfilter[1:Aindex]]
        newedges2 = [MyEdge((dorder[1], n)) for n in Bfilter[Aindex + 1:]]

        edges = newedges1 + newedges2

        return edges

    def add_road_segment_geo(self, edge, radius=1, epsilon=0.2):

        a, b, c1, c2 = self.__find_nodes_curbs(edge)
        m = mgh.bisect_angle(c1, b, c2, epsilon, radius=1)
        d1 = mgh.bisect_angle(a, b, m, epsilon, radius=radius)
        d2 = mgh.find_negative(d1, b)

        # figure out how the existing curb nodes connect to the new nodes
        new_d_edges = self.__find_d_connections(a, b, c1, c2, d1, d2)

        for e in new_d_edges:
            self.add_edge(e)

        # figure out how other involved parcels connect to the new nodes
        new_e_edges = self.__find_e_connections(a, b, c1, c2, d1, d2)

        for e in new_e_edges:
            self.add_edge(e)

        self.G.remove_node(b)
        self.roads_update = False

        return


# ###################################
#      PLOTTING FUNCTIONS
# ##################################

    def plot(self, **kwargs):
        plt.axes().set_aspect(aspect=1)
        plt.axis('off')
        edge_kwargs = kwargs.copy()
        nlocs = self.location_dict()
        edge_kwargs['label'] = "_nolegend"
        edge_kwargs['pos'] = nlocs
        nx.draw_networkx_edges(self.G, **edge_kwargs)
        node_kwargs = kwargs.copy()
        node_kwargs['label'] = self.name
        node_kwargs['pos'] = nlocs
        nodes = nx.draw_networkx_nodes(self.G, **node_kwargs)
        nodes.set_edgecolor('None')

    def plot_roads(self,
                   master=None,
                   update=False,
                   parcel_labels=False,
                   title="",
                   new_plot=True,
                   new_road_color="blue",
                   new_road_width=1.5,
                   old_node_size=25,
                   old_road_width=6,
                   barriers=True,
                   base_width=1,
                   stage=0):

        nlocs = self.location_dict()

        if update:
            self.define_roads()
            self.define_interior_parcels()

        if new_plot:
            plt.figure()

        edge_colors = [
            'blue' if e.road else
            'green' if e.barrier else 'red' if e.interior else 'black'
            for e in self.myedges()
        ]

        edge_width = [new_road_width * 2 if e.road
                      else 0.7*new_road_width if e.barrier
                      else 1.4*new_road_width if e.interior
                      else 1 for e in self.myedges()]

        node_colors = ['#008000' for n in self.G.nodes()]

        node_sizes = [1.4 for n in self.G.nodes()]

        # plot current graph
        nx.draw(self.G,
                         pos=nlocs,
                         with_labels=False,
                         node_size=node_sizes,
                        #  node_color=node_colors,
                         edge_color=edge_colors,
                         width=edge_width)

        # plot original roads
        if master:
            copy = master.copy()
            noffroad = [n for n in copy.G.nodes() if not n.road]
            for n in noffroad:
                copy.G.remove_node(n)
            eoffroad = [e for e in copy.myedges() if not e.road]
            for e in eoffroad:
                copy.G.remove_edge(e.nodes[0], e.nodes[1])

            # nx.draw_networkx(copy.G, pos=nlocs, with_labels=False,
            #                  node_size=old_node_size, node_color='black',
            #                  edge_color='black', width=old_road_width)

        # if stage==0:
        #     path='/data1/suhongyuan/road_planning/code/before.svg'
        # else:
        #     path='/data1/suhongyuan/road_planning/code/after.svg'
        # plt.savefig(path,format='svg',transparent=True)
        # plt.close()

    def plot_all_paths(self, all_paths, update=False):
        """ plots the shortest paths from all interior parcels to the road.
        Optional to update road geometery based on changes in network geometry.
        """

        plt.figure()
        if len(all_paths) == 0:
            self.plot_roads(update=update)
        else:
            Gs = []
            for p in all_paths:
                G = nx.subgraph(self.G, p)
                Gs.append(G)
            Gpaths = nx.compose_all(Gs, name="shortest paths")
            myGpaths = MyGraph(Gpaths)
            self.plot_roads(update=update)
            myGpaths.plot(edge_color='purple', width=6, node_size=1)

    def plot_weak_duals(self,
                        stack=None,
                        colors=None,
                        width=None,
                        node_size=None):
        """Given a list of weak dual graphs, plots them all. Has default colors
        node size, and line widths, but these can be added as lists."""

        if stack is None:
            duals = self.stacked_duals()
        else:
            duals = stack

        if colors is None:
            colors = [
                'grey', 'black', 'blue', 'purple', 'red', 'orange', 'yellow'
            ]
        else:
            colors = colors

        if width is None:
            width = [0.5, 0.75, 1, 1.75, 2.25, 3, 3.5]
        else:
            width = width

        if node_size is None:
            node_size = [0.5, 6, 9, 12, 17, 25, 30]
        else:
            node_size = node_size

        if len(duals) > len(colors):
            warnings.warn("too many dual graphs to draw. simplify fig," +
                          " or add more colors")

        plt.figure()

        for i in range(0, len(duals)):
            for j in duals[i]:
                j.plot(node_size=node_size[i],
                       node_color=colors[i],
                       edge_color=colors[i],
                       width=width[i])
                # print "color = {0}, node_size = {1}, width = {2}".format(
                #       colors[i], node_size[i], width[i])

        plt.axes().set_aspect(aspect=1)
        plt.axis('off')

if __name__ == "__main__":
    master = mgh.testGraphLattice(4)

    S0 = master.copy()

    S0.define_roads()
    S0.define_interior_parcels()

    road_edge = S0.myedges()[1]

    S0.add_road_segment(road_edge)

    S0.define_interior_parcels()

    mgh.test_dual(S0)

    plt.show()
