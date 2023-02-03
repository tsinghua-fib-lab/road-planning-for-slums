import numpy as np
from matplotlib import pyplot as plt
import shapefile
import math
from collections import defaultdict
import networkx as nx
import random
import itertools
import operator
from scipy.cluster.hierarchy import linkage, dendrogram
import json
import road_planning.envs.my_graph as mg

""" This file includes a bunch of helper functions for my_graph.py.
There are a bunch of basic spatial geometery functions,
some greedy search probablilty functions,
ways to set up and determine the shortest paths from parcel to a road
the code that exists on optimization problem 2: thinking about how to build in
additional connectivity beyond just minimum access, as well as plotting the
associated matrices
code for creating a mygraph object from a shapefile or a list of myfaces
(used for weak dual calculations)
a couple of test graphs- testGraph, (more or less lollipopo shaped) and
testGraphLattice which is a lattice.
"""

#############################
# BASIC MATH AND GEOMETRY FUNCTIONS
#############################


# myG geometry functions
def distance(mynode0, mynode1):
    return np.sqrt(distance_squared(mynode0, mynode1))


def distance_squared(mynode0, mynode1):
    return (mynode0.x - mynode1.x)**2 + (mynode0.y - mynode1.y)**2


def sq_distance_point_to_segment(target, myedge):
    """returns the square of the minimum distance between mynode
    target and myedge.   """
    n1 = myedge.nodes[0]
    n2 = myedge.nodes[1]

    if myedge.length == 0:
        sq_dist = distance_squared(target, n1)
    elif target == n1 or target == n2:
        sq_dist = 0
    else:
        px = float(n2.x - n1.x)
        py = float(n2.y - n1.y)
        u = float((target.x - n1.x) * px +
                  (target.y - n1.y) * py) / (px * px + py * py)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = n1.x + u * px
        y = n1.y + u * py

        dx = x - target.x
        dy = y - target.y

        sq_dist = (dx * dx + dy * dy)
    return sq_dist


def intersect(e1, e2):
    """ returns true if myedges e1 and e2 intersect """

    # fails for lines that perfectly overlap.
    def ccw(a, b, c):
        return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)

    a = e1.nodes[0]
    b = e1.nodes[1]
    c = e2.nodes[0]
    d = e2.nodes[1]

    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def are_parallel(e1, e2):
    """ returns true if myedges e1 and e2 are parallel """
    a = e1.nodes[0]
    b = e1.nodes[1]
    c = e2.nodes[0]
    d = e2.nodes[1]

    # check if parallel; handling divide by zero errors
    if a.x == b.x and c.x == d.x:  # check if both segments are flat
        parallel = True
    # if one is flat and other is not
    elif (a.x - b.x) * (c.x - d.x) == 0 and (a.x - b.x) + (c.x - d.x) != 0:
        parallel = False
    # if neither segment is flat and slopes are equal
    elif (a.y - b.y) / (a.x - b.x) == (c.y - d.y) / (c.x - d.x):
        parallel = True
    # n either segment is flat, slopes are not equal
    else:
        parallel = False
    return parallel


def segment_distance_sq(e1, e2):
    """returns the square of the minimum distance between myedges e1 and e2."""
    # check different
    if e1 == e2:
        sq_distance = 0
    # check parallel/colinear:
    # lines are not parallel/colinear and intersect
    if not are_parallel(e1, e2) and intersect(e1, e2):
        sq_distance = 0
    # lines don't intersect, aren't parallel
    else:
        d1 = sq_distance_point_to_segment(e1.nodes[0], e2)
        d2 = sq_distance_point_to_segment(e1.nodes[1], e2)
        d3 = sq_distance_point_to_segment(e2.nodes[0], e1)
        d4 = sq_distance_point_to_segment(e2.nodes[1], e1)
        sq_distance = min(d1, d2, d3, d4)

    return sq_distance


# vector math
def bisect_angle(a, b, c, epsilon=0.2, radius=1):
    """ finds point d such that bd bisects the lines ab and bc."""
    ax = a.x - b.x
    ay = a.y - b.y

    cx = c.x - b.x
    cy = c.y - b.y

    a1 = mg.MyNode(((ax, ay)) / np.linalg.norm((ax, ay)))
    c1 = mg.MyNode(((cx, cy)) / np.linalg.norm((cx, cy)))

    # if vectors are close to parallel, find vector that is perpendicular to ab
    # if they are not, then find the vector that bisects a and c
    if abs(np.cross(a1.loc, c1.loc)) < 0 + epsilon:
        # print("vectors {0}{1} and {1}{2} are close to //)".format(a,b,c)
        dx = -ay
        dy = ax
    else:
        dx = (a1.x + c1.x) / 2
        dy = (a1.y + c1.y) / 2

    # convert d values into a vector of length radius
    dscale = ((dx, dy) / np.linalg.norm((dx, dy))) * radius
    myd = mg.MyNode(dscale)

    # make d a node in space, not vector around b
    d = mg.MyNode((myd.x + b.x, myd.y + b.y))

    return d


def find_negative(d, b):
    """finds the vector -d when b is origen """
    negx = -1 * (d.x - b.x) + b.x
    negy = -1 * (d.y - b.y) + b.y
    dneg = mg.MyNode((negx, negy))
    return dneg


# clean up and probability functions
def WeightedPick(d):
    """picks an item out of the dictionary d, with probability proportional to
    the value of that item.  e.g. in {a:1, b:0.6, c:0.4} selects and returns
    "a" 5/10 times, "b" 3/10 times and "c" 2/10 times. """

    r = random.uniform(0, sum(d.values()))
    s = 0.0
    for k, w in d.items():
        s += w
        if r < s:
            return k
    return k


def mat_reorder(matrix, order):
    """sorts a square matrix so both rows and columns are
    ordered by order. """

    Drow = [matrix[i] for i in order]
    Dcol = [[r[i] for i in order] for r in Drow]

    return Dcol


def myRoll(mylist):
    """rolls a list, putting the last element into the first slot. """

    mylist.insert(0, mylist[-1])
    del mylist[-1]
    return (mylist)


######################
# DUALS HElPER
#######################


def form_equivalence_classes(myG, only_k_max=True,duals=None, verbose=False):

    try:
        for f in myG.inner_facelist:
            f.even_nodes = {}
            f.odd_node = {}
    except:
        pass

    if not duals:
        duals = myG.stacked_duals()

    if only_k_max:
        result={}
        return result,len(duals)

    depth = 1
    result = {}

    myG.S1_nodes()
    result[depth] = [f for f in myG.inner_facelist if f.odd_node[depth]]

    if verbose:
        # print("Graph S{} has {} parcels".format(depth, len(result[depth])))
        pass

    depth += 1

    if verbose:
        test_interior_is_inner(myG)

    while depth < len(duals):
        duals, depth, result = myG.formClass(duals, depth, result)
        if verbose:
            # md = max(result.keys())
            # print("Graph S{} has {} parcels".format(md, len(result[md])))
            # print("current depth {} just finished".format(depth))
            # test_interior_is_inner(myG)
            pass

    return result, depth


######################
# DEALING WITH PATHS
#######################


def ptup_to_mypath(myG, ptup):
    mypath = []

    for i in range(1, len(ptup)):
        pedge = myG.G[ptup[i - 1]][ptup[i]]['myedge']
        mypath.append(pedge)

    return mypath


def path_length(path):
    """finds the geometric path length for a path that consists of a list of
    MyNodes. """
    length = 0
    for i in range(1, len(path)):
        length += distance(path[i - 1], path[i])
    return length


# def path_length_npy(path):
#    xy = np.array([n.x,n.y for n in path])
#    return np.linalg.norm(xy[1:] - xy[:-1],2,1).sum()


def shorten_path(ptup):
    """ all the paths found in my pathfinding algorithm start at the fake
    road side, and go towards the interior of the parcel.  This method drops
    nodes beginning at the fake road node, until the first and only the
    first node is on a road.  This gets rid of paths that travel along a
    curb before ending."""

    while ptup[1].road is True and len(ptup) > 2:
        ptup.pop(0)
    return ptup


def segment_near_path(myG, segment, pathlist, threshold):
    """returns True if the segment is within (geometric) distance threshold
    of all the segments contained in path is stored as a list of nodes that
    strung together make up a path.
    """
    # assert isinstance(segment, mg.MyEdge)

    # pathlist = ptup_to_mypath(path)

    for p in pathlist:
        sq_distance = segment_distance_sq(p, segment)
        if sq_distance < threshold**2:
            return True

    return False


def _fake_edge(myA, centroid, mynode):
    newedge = mg.MyEdge((centroid, mynode))
    newedge.length = 0
    myA.add_edge(newedge)


def __add_fake_edges(myA, p, roads_only=False):
    if roads_only:
        [_fake_edge(myA, p.centroid, n) for n in p.nodes if n.road]
    else:
        [_fake_edge(myA, p.centroid, n) for n in p.nodes]


def shortest_path_setup(myA, p, roads_only=False):
    """ sets up graph to be ready to find the shortest path from a
    parcel to the road. if roads_only is True, only put fake edges for the
    interior parcel to nodes that are already connected to a road. """

    fake_interior = p.centroid

    __add_fake_edges(myA, p)

    fake_road_origin = mg.MyNode((305620, 8022470))

    for i in myA.road_nodes:
        if len(myA.G.neighbors(i)) > 2:
            _fake_edge(myA, fake_road_origin, i)
    return fake_interior, fake_road_origin


def shortest_path_p2p(myA, p1, p2):
    """finds the shortest path along fencelines from a given interior parcel
    p1 to another parcel p2"""

    __add_fake_edges(myA, p1, roads_only=True)
    __add_fake_edges(myA, p2, roads_only=True)

    path = nx.shortest_path(myA.G, p1.centroid, p2.centroid, "weight")
    length = nx.shortest_path_length(myA.G, p1.centroid, p2.centroid, "weight")

    myA.G.remove_node(p1.centroid)
    myA.G.remove_node(p2.centroid)

    return path[1:-1], length


def find_short_paths(myA, parcel, barriers=True, shortest_only=False):
    """ finds short paths from an interior parcel,
    returns them and stores in parcel.paths  """

    rb = [n for n in parcel.nodes + parcel.edges if n.road]
    if len(rb) > 0:
        raise AssertionError("parcel %s is on a road") % (str(parcel))

    if barriers:
        barrier_edges = [e for e in myA.myedges() if e.barrier]
        if len(barrier_edges) > 0:
            myA.remove_myedges_from(barrier_edges)
        else:
            print("no barriers found. Did you expect them?")
        # myA.plot_roads(title = "myA no barriers")

    interior, road = shortest_path_setup(myA, parcel)

    shortest_path = nx.shortest_path(myA.G, road, interior, "weight")
    if shortest_only is False:
        shortest_path_segments = len(shortest_path)
        shortest_path_distance = path_length(shortest_path[1:-1])
        all_simple = [
            shorten_path(p[1:-1]) for p in nx.all_simple_paths(
                myA.G, road, interior, cutoff=shortest_path_segments + 2)
        ]
        paths = dict((tuple(p), path_length(p)) for p in all_simple
                     if path_length(p) < shortest_path_distance * 2)
    if shortest_only is True:
        p = shorten_path(shortest_path[1:-1])
        paths = {tuple(p): path_length(p)}

    myA.G.remove_node(road)
    myA.G.remove_node(interior)
    if barriers:
        for e in barrier_edges:
            myA.add_edge(e)

    parcel.paths = paths

    return paths


def find_short_paths_all_parcels(myA,
                                 flist=None,
                                 full_path=None,
                                 barriers=True,
                                 quiet=False,
                                 shortest_only=False):
    """ finds the short paths for all parcels, stored in parcel.paths
    default assumes we are calculating from the outside in.  If we submit an
    flist, find the parcels only for those faces, and (for now) recaluclate
    paths for all of those faces.
    """
    all_paths = {}
    counter = 0

    if flist is None:
        flist = myA.interior_parcels

    for parcel in flist:
        # if paths have already been defined for this parcel
        # (full path should exist too)
        if parcel.paths:

            if full_path is None:
                raise AssertionError("comparison path is None "
                                     "but parcel has paths")

            rb = [n for n in parcel.nodes + parcel.edges if n.road]
            if len(rb) > 0:
                raise AssertionError("parcel %s is on a road" % (parcel))

            needs_update = False
            for pathitem in parcel.paths.items():
                path = pathitem[0]
                mypath = ptup_to_mypath(myA, path)
                path_length = pathitem[1]
                for e in full_path:
                    if segment_near_path(myA, e, mypath, path_length):
                        needs_update = True
                        # this code would be faster if I could break to
                        # next parcel if update turned true.
                        break

            if needs_update is True:
                paths = find_short_paths(myA,
                                         parcel,
                                         barriers=barriers,
                                         shortest_only=shortest_only)
                counter += 1
                all_paths.update(paths)
            elif needs_update is False:
                paths = parcel.paths
                all_paths.update(paths)
        # if paths have not been defined for this parcel
        else:
            paths = find_short_paths(myA,
                                     parcel,
                                     barriers=barriers,
                                     shortest_only=shortest_only)
            counter += 1
            all_paths.update(paths)
    if quiet is False:
        pass
        # print("Shortest paths found for {} parcels".format(counter))

    return all_paths


def build_path(myG, start, finish):
    ptup = nx.shortest_path(myG.G, start, finish, weight="weight")

    ptup = shorten_path(ptup)
    ptup.reverse()
    ptup = shorten_path(ptup)

    mypath = ptup_to_mypath(myG, ptup)

    for e in mypath:
        myG.add_road_segment(e)

    return ptup, mypath


#############################################
#  PATH SELECTION AND CONSTRUCTION
#############################################
def find_all_one_road(myG):
    elist=[e for e in myG.G.edges() if ((e[0].road and not e[1].road) or (e[1].road and not e[0].road))]
    # print(elist)
    return elist

def choose_path(myG, paths, alpha, random_road=False, strict_greedy=False):
    """ chooses the path segment, choosing paths of shorter
    length more frequently  """
    if random_road:
        target_path = random.choice(paths)
    else:
        if strict_greedy is False:
            inv_weight = dict((k, 1.0 / (paths[k]**alpha)) for k in paths)
            target_path = WeightedPick(inv_weight)
        if strict_greedy is True:
            target_path = min(paths, key=paths.get)

    mypath = ptup_to_mypath(myG, target_path)

    return target_path, mypath


#        if outsidein:
#            result, depth = form_equivalence_classes(myG)
#            while len(flist) < 1:
#                md = max(result.keys())
#                flist = flist + result.pop(md)
#        elif outsidein == False:
#            flist = myG.interior_parcels
#            ## alternate option:
#            # result, depth = form_equivalence_classes(myG)
#            # flist = result[3]


def build_all_roads(myG,
                    master=None,
                    alpha=8,
                    road_max=None,
                    plot_intermediate=False,
                    wholepath=True,
                    original_roads=None,
                    plot_original=False,
                    bisect=False,
                    plot_result=False,
                    barriers=False,
                    quiet=False,
                    vquiet=False,
                    random_road=False,
                    strict_greedy=False,
                    outsidein=False):
    """builds roads using the probablistic greedy alg, until all
    interior parcels are connected, and returns the total length of
    road built. """

    if vquiet is True:
        quiet = True

    if plot_original:
        myG.plot_roads(original_roads,
                       update=False,
                       parcel_labels=False,
                       new_road_color="blue")

    shortest_only = False
    if strict_greedy is True:
        shortest_only = True

    added_road_length = 0
    plotnum = 0
    if plot_intermediate is True and master is None:
        master = myG.copy()

    myG.define_interior_parcels()

    target_mypath = None
    if vquiet is False:
        print("Begin w {} Int Parcels".format(len(myG.interior_parcels)))

    # before the max depth (md) is calculated, just assume it's very large in
    # in order ensure we find the equivalence classes at least once.
    md = 100
    flag=True
    road_num=0

    k_time=[]
    total_time=[]

    while (myG.interior_parcels or road_max):

        if road_max:
            if road_num>=road_max:
                break
    
        if not myG.interior_parcels and flag or road_num==10:
            myG.plot_roads(master, update=False)
            plt.savefig("Int_Step" + str(road_num) + ".pdf", format='pdf')
            plotnum += 1
            if road_num != 10:
                flag=False

        start_w = time.clock()

        # start = time.clock()
        # result, depth = form_equivalence_classes(myG,only_k_max=random_road)
        # end = time.clock()
        # k_time.append(end-start)
        # print("depth:%s"%depth)
        # print("k_time: %s" % (end - start))

        if not random_road:
        # flist from result!
            flist = []

            if md == 3:
                flist = myG.interior_parcels
            elif md > 3:
                if outsidein is False:
                    result, depth = form_equivalence_classes(myG,only_k_max=random_road)
                    while len(flist) < 1:
                        md = max(result.keys())
                        flist = flist + result.pop(md)
                elif outsidein is True:
                    result, depth = form_equivalence_classes(myG,only_k_max=random_road)
                    md = max(result.keys())
                    if len(result[md]) == 0:
                        md = md - 2
                    flist = list(set(result[3]) - set(result.get(5, [])))

        if quiet is False:
            pass

        if random_road is True:
            all_paths = find_all_one_road(myG)
            road_num +=1
        else:
            # potential segments from parcels in flist
            all_paths = find_short_paths_all_parcels(myG,
                                                    flist,
                                                    target_mypath,
                                                    barriers,
                                                    quiet=quiet,
                                                    shortest_only=shortest_only)
            # choose and build one

        target_ptup, target_mypath = choose_path(myG,
                                                all_paths,
                                                alpha,
                                                random_road=random_road,
                                                strict_greedy=strict_greedy)

        if wholepath is False:
            added_road_length += target_mypath[0].length
            myG.add_road_segment(target_mypath[0])

        if wholepath is True:
            for e in target_mypath:
                added_road_length += e.length
                myG.add_road_segment(e)

        myG.define_interior_parcels()

        if plot_intermediate:
            myG.plot_roads(master, update=False)
            plt.savefig("Int_Step" + str(plotnum) + ".pdf", format='pdf')
            plotnum += 1

        remain = len(myG.interior_parcels)
        end_w = time.clock()
        total_time.append(end_w-start_w)
        print("total_time: %s" % (end_w - start_w))

        if quiet is False:
            print("\n{} interior parcels left".format(remain))
        if vquiet is False:
            if remain > 300 or remain in [50, 100, 150, 200, 225, 250, 275]:
                print("{} interior parcels left".format(remain))

    # update the properties of nodes & edges to reflect new geometry.
    # print("avr_k_time:%s"%(sum(k_time)/len(k_time)))
    # print("avr_total_time:%s"%(sum(total_time)/len(total_time)))
    myG.added_roads = added_road_length
    return added_road_length


def bisecting_road(myG):
    # once done getting all interior parcels connected, have option to bisect
    bisecting_roads = 0

    start, finish = bisecting_path_endpoints(myG)
    ptup, myedges = build_path(myG, start, finish)
    bisecting_roads = path_length(ptup)

    myG.added_roads = myG.added_roads + bisecting_roads
    return bisecting_roads


############################
# connectivity optimization
############################


def __road_connections_through_culdesac(myG, threshold=5):
    """connects all nodes on a road that are within threshold = 5 meters of
    each other.  This means that paths can cross a culdesac instead of needing
    to go around. """

    etup_drop = []

    nlist = [i for i in myG.G.nodes() if i.road is True]
    for i, j in itertools.combinations(nlist, 2):
        if j in myG.G and i in myG.G:
            if distance(i, j) < threshold:
                newE = mg.MyEdge((i, j))
                newE.road = True
                myG.add_edge(newE)
                etup_drop.append((i, j))

    return etup_drop


def shortest_path_p2p_matrix(myG, full=False, travelcost=False):
    """option if full is false, removes all interior edges, so that travel
    occurs only along a road.  If full is true, keeps interior edges.  If
    travel cost is true, increases weight of non-road edges by a factor of ten.
    Base case is defaults of false and false."""

    copy = myG.copy()

    etup_drop = copy.find_interior_edges()
    if full is False:
        copy.G.remove_edges_from(etup_drop)
        # print("dropping {} edges".format(len(etup_drop)))

    __road_connections_through_culdesac(copy)

    path_mat = []
    path_len_mat = []

    edges = copy.myedges()

    if travelcost is True:
        for e in edges:
            if e.road is False:
                copy.G[e.nodes[0]][e.nodes[1]]['weight'] = e.length * 10

    for p0 in copy.inner_facelist:
        path_vec = []
        path_len_vec = []
        __add_fake_edges(copy, p0)

        for p1 in copy.inner_facelist:
            if p0.centroid == p1.centroid:
                length = 0
                path = p0.centroid
            else:
                __add_fake_edges(copy, p1)
                try:
                    path = nx.shortest_path(copy.G, p0.centroid, p1.centroid,
                                            "weight")
                    length = path_length(path[1:-1])
                except:
                    path = []
                    length = np.nan
                copy.G.remove_node(p1.centroid)
            path_vec.append(path)
            path_len_vec.append(length)

        copy.G.remove_node(p0.centroid)
        path_mat.append(path_vec)
        path_len_mat.append(path_len_vec)

        n = len(path_len_mat)
        meantravel = (sum([sum(i) for i in path_len_mat]) / (n * (n - 1)))

    return path_len_mat, meantravel


def difference_roads_to_fences(myG, travelcost=False):
    fullpath_len, tc = shortest_path_p2p_matrix(myG, full=True)
    path_len, tc = shortest_path_p2p_matrix(myG, full=False)

    diff = [[
        path_len[j][i] - fullpath_len[j][i]
        for i in range(0, len(fullpath_len[j]))
    ] for j in range(0, len(fullpath_len))]

    dmax = max([max(i) for i in diff])
    # print dmax

    for j in range(0, len(path_len)):
        for i in range(0, len(path_len[j])):
            if np.isnan(path_len[j][i]):
                diff[j][i] = dmax + 150
                #            diff[j][i] = np.nan

    n = len(path_len)
    meantravel = sum([sum(i) for i in path_len]) / (n * (n - 1))

    return diff, fullpath_len, path_len, meantravel


def bisecting_path_endpoints(myG):
    roads_only = myG.copy()
    etup_drop = roads_only.find_interior_edges()
    roads_only.G.remove_edges_from(etup_drop)
    __road_connections_through_culdesac(roads_only)
    # nodes_drop = [n for n in roads_only.G.nodes() if not n.road]
    # roads_only.G.remove_nodes_from(nodes_drop)

    distdict = {}

    for i, j in itertools.combinations(myG.road_nodes, 2):
        geodist_sq = distance_squared(i, j)
        onroad_dist = nx.shortest_path_length(roads_only.G,
                                              i,
                                              j,
                                              weight='weight')
        dist_diff = onroad_dist**2 / geodist_sq
        distdict[(i, j)] = dist_diff
    (i, j) = max(distdict.iteritems(), key=operator.itemgetter(1))[0]

    return i, j


################
# GRAPH INSTANTIATION
###################


def graphFromMyEdges(elist, name=None):
    myG = mg.MyGraph(name=name)
    for e in elist:
        myG.add_edge(e)
    return myG


def graphFromMyFaces(flist, name=None):
    myG = mg.MyGraph(name=name)
    for f in flist:
        for e in f.edges:
            myG.add_edge(e)
    return myG


def graphFromShapes(shapes, name, rezero=np.array([0, 0])):
    nodedict = dict()
    plist = []
    for s in shapes:
        nodes = []
        for k in s.points:
            k = k - rezero
            myN = mg.MyNode(k)
            if myN not in nodedict:
                nodes.append(myN)
                nodedict[myN] = myN
            else:
                nodes.append(nodedict[myN])
            edges = [(nodes[i], nodes[i + 1])
                     for i in range(0,
                                    len(nodes) - 1)]
            plist.append(mg.MyFace(edges))

    myG = mg.MyGraph(name=name)

    for p in plist:
        for e in p.edges:
            myG.add_edge(mg.MyEdge(e.nodes))

    return myG


def is_roadnode(node, graph):
    """defines a node as a road node if any connected edges are road edges.
    returns true or false and updates the properties of the node. """
    graph.G[node].keys()
    node.road = False
    for k in graph.G[node].keys():
        edge = graph.G[node][k]['myedge']
        if edge.road is True:
            node.road = True
            return node.road
    return node.road


def is_interiornode(node, graph):
    """defines a node as an interior node if any connected edges are interior
    edges. returns true or false and updates the properties of the node. """
    graph.G[node].keys()
    node.interior = False
    for k in graph.G[node].keys():
        edge = graph.G[node][k]['myedge']
        if edge.interior is True:
            node.interior = True
            return node.interior
    return node.interior


def is_barriernode(node, graph):
    """defines a node as a road node if any connected edges are barrier edges.
    returns true or false and updates the properties of the node. """
    graph.G[node].keys()
    node.barrier = False
    for k in graph.G[node].keys():
        edge = graph.G[node][k]['myedge']
        if edge.barrier is True:
            node.barrier = True
            return node.barrier
    return node.barrier


def graphFromJSON(jsonobj):
    """returns a new mygraph from a json object.  calculates interior node
    and graph properties from the properties of the edges.
    """

    edgelist = []
    # read all the edges from json
    for feature in jsonobj['features']:
        # check that there are exactly 2 nodes
        numnodes = len(feature['geometry']['coordinates'])
        if numnodes != 2:
            raise AssertionError("JSON line feature has {} "
                                 "coordinates instead of 2".format(numnodes))

        c0 = feature['geometry']['coordinates'][0]
        c1 = feature['geometry']['coordinates'][1]

        isinterior = feature['properties']['interior']
        isroad = feature['properties']['road']
        isbarrier = feature['properties']['barrier']

        n0 = mg.MyNode(c0)
        n1 = mg.MyNode(c1)

        edge = mg.MyEdge((n0, n1))
        edge.road = json.loads(isroad)
        edge.interior = json.loads(isinterior)
        edge.barrier = json.loads(isbarrier)
        edgelist.append(edge)

    # create a new graph from the edge list, and calculate
    # necessary graph properties from the road
    new = graphFromMyEdges(edgelist)
    new.road_edges = [e for e in new.myedges() if e.road]
    new.road_nodes = [n for n in new.G.nodes() if is_roadnode(n, new)]
    new.interior_nodes = [n for n in new.G.nodes() if is_interiornode(n, new)]
    new.barrier_nodes = [n for n in new.G.nodes() if is_barriernode(n, new)]

    # defines all the faces in the graph
    new.inner_facelist
    # defines all the faces with no road nodes in the graph as interior parcels
    new.define_interior_parcels()

    return new, edgelist


def import_and_setup(filename,
                     threshold=1,
                     component=None,
                     rezero=np.array([0, 0]),
                     byblock=True,
                     name=""):
    """ threshold defines the minimum distance (in map units) between two nodes
    before they are combined into a single node during the clean up phase. This
    helps to handle poorly written polygon geometery.

    Component is an option that lets you return a single block (they're ordered
    by number of nodes, where 0 is the largest) instead of all of the blocks in
    the map.

    byblock = True runs the clean up geometery procedure on each original
    block individually, rather than all the blocks together.  This makes the
    clean up process a lot faster for large numbers of blocks, but if there are
    pieces of a block that are supposed to be connected, but are not in the
    original map.
    """

    # plt.close('all')

    # check that rezero is an array of len(2)
    # check that threshold is a float

    sf = shapefile.Reader(filename)
    myG1 = graphFromShapes(sf.shapes(), name, rezero)

    print("shape file loaded")

    myG1 = myG1.clean_up_geometry(threshold, err, byblock)

    print("geometery cleaned up")

    xmin = min([n.x for n in myG1.G.nodes()])
    ymin = min([n.y for n in myG1.G.nodes()])

    rezero_vector = np.array([xmin, ymin])
    rescale_vector = np.ones(2)/max(myG1.myweight())

    myG2 = rescale_mygraph(myG1, rezero=rezero_vector, rescale=rescale_vector)
    myG2.rezero_vector = rezero_vector
    myG2.rescale_vector = rescale_vector

    if component is None:
        return myG2
    else:
        return myG2.connected_components()[component]


def rescale_mygraph(myG, rezero=np.array([0, 0]), rescale=np.array([1, 1])):
    """returns a new graph (with no interior properties defined), rescaled under
    a linear function newloc = (oldloc-rezero)*rescale  where all of those are
    (x,y) numpy arrays.  Default of rezero = (0,0) and rescale = (1,1) means
    the locations of nodes in the new and old graph are the same.
    """

    scaleG = mg.MyGraph()
    for e in myG.myedges():
        n0 = e.nodes[0]
        n1 = e.nodes[1]
        nn0 = mg.MyNode((n0.loc - rezero) * rescale)
        nn1 = mg.MyNode((n1.loc - rezero) * rescale)
        scaleG.add_edge(mg.MyEdge((nn0, nn1)))

    return scaleG


def build_barriers(barriers):

    for b in barriers:
        b.barrier = True
        b.road = False
        for n in b.nodes:
            n.barrier = True
            n.road = False


####################
# Testing functions
###################


def test_edges_equality():
    """checks that myGraph points to myEdges correctly   """
    testG = testGraph()
    testG.trace_faces()
    outerE = list(testG.outerface.edges)[0]
    return outerE is testG.G[outerE.nodes[0]][outerE.nodes[1]]['myedge']


def test_dual(myG):
    """ plots the weak duals based on testGraph"""

    S0 = myG.weak_dual()

    myG.plot_roads(update=False)
    S0.plot(node_color='g', edge_color='g', width=3)


def test_nodes(n1, n2):
    """ returns true if two nodes are evaluated as the same"""
    eq_num = len(set(n1).intersection(set(n2)))
    is_num = len(
        set([id(n) for n in n1]).intersection(set([id(n) for n in n2])))
    print("is eq? ", eq_num, "is is? ", is_num)


def test_interior_is_inner(myG):
    myG.inner_facelist
    myG.interior_parcels
    in0 = myG.interior_parcels[0]
    ans = in0 in myG.inner_facelist

    # print("interior in inner_facelist is {}".format(ans))

    return ans


def testGraph():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((0, 2))
    n[4] = mg.MyNode((0, 3))
    n[5] = mg.MyNode((1, 2))
    n[6] = mg.MyNode((1, 3))
    n[7] = mg.MyNode((0, 4))
    n[8] = mg.MyNode((-1, 4))
    n[9] = mg.MyNode((-1, 3))
    n[10] = mg.MyNode((-1, 2))
    n[11] = mg.MyNode((1, 4))
    n[12] = mg.MyNode((-2, 3))

    lat = mg.MyGraph(name="S0")
    lat.add_edge(mg.MyEdge((n[1], n[2])))
    lat.add_edge(mg.MyEdge((n[2], n[3])))
    lat.add_edge(mg.MyEdge((n[2], n[5])))
    lat.add_edge(mg.MyEdge((n[3], n[4])))
    lat.add_edge(mg.MyEdge((n[3], n[5])))
    lat.add_edge(mg.MyEdge((n[3], n[9])))
    lat.add_edge(mg.MyEdge((n[4], n[5])))
    lat.add_edge(mg.MyEdge((n[4], n[6])))
    lat.add_edge(mg.MyEdge((n[4], n[7])))
    lat.add_edge(mg.MyEdge((n[4], n[8])))
    lat.add_edge(mg.MyEdge((n[4], n[9])))
    lat.add_edge(mg.MyEdge((n[5], n[6])))
    lat.add_edge(mg.MyEdge((n[6], n[7])))
    lat.add_edge(mg.MyEdge((n[7], n[8])))
    lat.add_edge(mg.MyEdge((n[8], n[9])))
    lat.add_edge(mg.MyEdge((n[9], n[10])))
    lat.add_edge(mg.MyEdge((n[3], n[10])))
    lat.add_edge(mg.MyEdge((n[2], n[10])))
    lat.add_edge(mg.MyEdge((n[7], n[11])))
    lat.add_edge(mg.MyEdge((n[6], n[11])))
    lat.add_edge(mg.MyEdge((n[10], n[12])))
    lat.add_edge(mg.MyEdge((n[8], n[12])))

    return lat


def testGraphLattice(n, xshift=0, yshift=0, scale=1):
    """returns a square lattice of dimension nxn   """
    nodelist = {}
    for j in range(0, n**2):
        x = (math.fmod(j, n)) * scale + xshift
        y = (math.floor(j / n)) * scale + yshift
        nodelist[j] = mg.MyNode((x, y))

    edgelist = defaultdict(list)

    for i in nodelist.keys():
        ni = nodelist[i]
        for j in nodelist.keys():
            nj = nodelist[j]
            if ni != nj:
                if distance(ni, nj) == scale:
                    edgelist[ni].append(nj)

    myedgelist = []

    for n1 in edgelist.keys():
        n2s = edgelist[n1]
        for n2 in n2s:
            myedgelist.append(mg.MyEdge((n1, n2)))

    lattice = graphFromMyEdges(myedgelist)
    lattice.name = "lattice"

    return lattice


def testGraphEquality():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((1, 1))
    n[4] = mg.MyNode((1, 0))
    n[5] = mg.MyNode((0, 0))  # actually equal
    n[6] = mg.MyNode((0.0001, 0.0001))  # within rounding
    n[7] = mg.MyNode((0.1, 0.1))  # within threshold
    n[8] = mg.MyNode((0.3, 0.3))  # actually different

    G = mg.MyGraph(name="S0")
    G.add_edge(mg.MyEdge((n[1], n[2])))
    G.add_edge(mg.MyEdge((n[2], n[3])))
    G.add_edge(mg.MyEdge((n[3], n[4])))
    G.add_edge(mg.MyEdge((n[4], n[5])))
    G.add_edge(mg.MyEdge((n[5], n[6])))
    G.add_edge(mg.MyEdge((n[6], n[7])))
    G.add_edge(mg.MyEdge((n[7], n[8])))

    return G, n


def json_test(test_geojson):
    """  If the good geoJSON request does not show an OK status message, the
    validation server is down.  """

    validate_endpoint = 'http://geojsonlint.com/validate'
    good_geojson = '{"type": "Point", "coordinates": [-100, 80]}'
    good_request = requests.post(validate_endpoint, data=good_geojson)
    test_request = requests.post(validate_endpoint, data=test_geojson)

    print("hard coded good geoJSON:")
    print(good_request.json())
    print("status for test geojson:")
    print(test_request.json())


def __centroid_test():
    n = {}
    n[1] = mg.MyNode((0, 0))
    n[2] = mg.MyNode((0, 1))
    n[3] = mg.MyNode((1, 1))
    n[4] = mg.MyNode((1, 0))
    n[5] = mg.MyNode((0.55, 0))
    n[6] = mg.MyNode((0.5, 0.9))
    n[7] = mg.MyNode((0.45, 0))
    n[8] = mg.MyNode((0.4, 0))
    n[9] = mg.MyNode((0.35, 0))
    n[10] = mg.MyNode((0.3, 0))
    n[11] = mg.MyNode((0.25, 0))
    nodeorder = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1]
    nodetups = [(n[nodeorder[i]], n[nodeorder[i + 1]])
                for i in range(0,
                               len(nodeorder) - 1)]
    edgelist = [mg.MyEdge(i) for i in nodetups]

    f1 = mg.MyFace(nodetups)
    S0 = graphFromMyFaces([f1])

    S0.define_roads()
    S0.define_interior_parcels()

    S0.plot_roads(parcel_labels=True)

    return S0, f1, n, edgelist


def testmat():
    testmat = []
    dim = 4
    for i in range(0, dim):
        k = []
        for j in range(0, dim):
            k.append((i - j) * (i - j))
        testmat.append(k)
    return testmat


def build_lattice_barrier(myG):
    edgesub = [
        e for e in myG.myedges() if e.nodes[0].y == 0 and e.nodes[1].y == 0
    ]
    # barrieredges = [e for e in edgesub if e.nodes[1].y == 0]

    for e in edgesub:
        myG.remove_road_segment(e)
        e.barrier = True

    myG.define_interior_parcels()
    return myG, edgesub


if __name__ == "__main__":
    master = testGraphLattice(7)
    master.name = "Lat_0"
    master.define_roads()
    master.define_interior_parcels()
    # S0, barrier_edges = build_lattice_barrier(S0)
    # barGraph = graphFromMyEdges(barrier_edges)
    S0 = master.copy()

    # S0.plot_roads(master, update=False, new_plot=True)

    test_dual(S0)

    S0 = master.copy()
    new_roads_i = build_all_roads(S0,
                                  master,
                                  alpha=2,
                                  wholepath=True,
                                  barriers=False,
                                  plot_intermediate=False,
                                  strict_greedy=True,
                                  vquiet=True,
                                  outsidein=True)

    S0.plot_roads()
    print("outside to in" + str(new_roads_i))

    S0 = master.copy()
    new_roads_i = build_all_roads(S0,
                                  master,
                                  alpha=2,
                                  wholepath=True,
                                  barriers=False,
                                  plot_intermediate=True,
                                  strict_greedy=True,
                                  vquiet=True,
                                  outsidein=False)

    S0.plot_roads()
    print("inside out" + str(new_roads_i))
