from matplotlib import pyplot as plt
import pickle
import pretreatment.my_graph_helpers as mgh
"""
This example file demonstrates how to import a shapefile, finding and
 plotting the shortest distance of roads necessary to acheive universal road
 access for all parcels.

The shortest distance of roads algorithm is based on probablistic greedy search
so different runs will give slightly different answers.

 """


def new_import(filename, name=None, byblock=True, threshold=1,err=0.2,xmin=-100,xmax=100,ymin=-100,ymax=100):
    """ imports the file, plots the original map, and returns
    a list of blocks from the original map.
    """

    if name is None:
        name = filename 

    original = mgh.import_and_setup(filename,
                                    threshold=threshold,
                                    err=err,
                                    byblock=byblock,
                                    name=name,xmin=xmin,xmax=xmax,ymin=ymin,ymax=ymax)

    blocklist = original.connected_components()

    print("This map has {} block(s). \n".format(len(blocklist)))

    # plt.figure()
    # # plot the full original map
    # for b in blocklist:
    #     # defines original geometery as a side effect,
    #     b.plot_roads(master=b, new_plot=False, update=True)

    blocklist.sort(key=lambda b: len(b.myedges()), reverse=True)

    return blocklist


def run_once(blocklist):
    """Given a list of blocks, builds roads to connect all interior parcels and
    plots all blocks in the same figure.
    """

    map_roads = 0
    plt.figure()

    for original in blocklist:
        original.define_roads()
        original.define_interior_parcels()
        if len(original.interior_parcels) > 0:
            block = original.copy()

            # define interior parcels in the block based on existing roads
            # block.define_interior_parcels()

            # finds roads to connect all interior parcels for a given block
            block_roads = mgh.build_all_roads(block, road_max=1,random_road=True,wholepath=True)
            map_roads = map_roads + block_roads
        else:
            block = original.copy()

        block.plot_roads(master=original, new_plot=False)

    return map_roads ,block


if __name__ == "__main__":

    # SINGLE SMALL BLOCK
    # filename = "data/Epworth_Demo"
    # name = "ep single"
    # byblock = False
    # threshold = 5.5
    # err=3.15

    # MANY SMALL BLOCKS, epworth
    # # some of the blocks here require a threshold of 0.5
    # filename = "data/Epworth_Before"
    # name = "ep many"
    # byblock = False
    # threshold = 8
    # err=0.2

    # MANY SMALL BLOCKS, Phule Nagar
    # some of the blocks here require a threshold of 0.5
    # filename = "data/Phule_Nagar_v6"
    # name = "phule"
    # byblock = False
    # threshold = 2
    # err=0.2

    # ONE LARGE BLOCK
    # filename = "data/CapeTownSouth"
    # name = "cape"
    # byblock = False
    # threshold = 2
    # err=3.15

    # filename = 'data/CapeTown_zx'
    # byblock = False
    # name='cape'
    # err=3.15
    # # threshold = 2.0
    # threshold = 0.5

    # filename = 'data/CapeTown_zs'
    # byblock = False
    # name='cape'
    # err=3.15
    # # threshold = 2.0
    # threshold = 0.5

    filename = 'india'
    mappath = 'pretreatment/data/' + filename
    byblock = False
    name='cape'
    err=3.15
    # threshold = 1.5
    threshold=1.0

    blocklist = new_import(mappath,
                           name,
                           byblock=byblock,
                           threshold=threshold,
                           err=err)

    g0 = blocklist[0]
    g0.define_roads()
    g0.define_interior_parcels()
    eall=g0.myedges()
    wall=g0.myweight()
    g0.plot_roads(stage=1)

    with open('data/{}.mg'.format(filename),'wb') as mgfile:
        mgfile.write(pickle.dumps(g0))
        mgfile.close

    print(len(g0.inner_facelist))
    print(len(g0.interior_parcels))
    print(len(g0.myedges()))

    g0.plot_roads(new_plot=False)
    plt.show()
    # g1 = blocklist[1]
    # g1.define_roads()
    # g1.define_interior_parcels()
    # with open('Save/{}_less.mg'.format(filename),'wb') as mgfile:
    #     mgfile.write(pickle.dumps(g1))
    #     mgfile.close
    # g1.plot_roads(new_plot=False)
    # plt.show()

    # g.save_graph('Save\{}.shp'.format(filename))
    # with open('Save/{}.mg'.format(filename),'wb') as mgfile:
    #     mgfile.write(pickle.dumps(g0))
    #     mgfile.close



# 
    # ep_geojson = g.myedges_geoJSON()
    # map_roads, newblock = run_once([blocklist[0]])

