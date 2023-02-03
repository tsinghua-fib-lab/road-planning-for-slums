from matplotlib import pyplot as plt

import my_graph_helpers as mgh
import math
"""
This shows a short snippet of code to (manually) define barriers for the
epworth block, build the barriers through the function "build_barriers", and
plot the results before and after construction of new roads.

 """


def define_epworth_barriers(myG):

    be = [
        e for e in myG.myedges() if e.nodes[0].x > 182 and e.nodes[1].x > 182
    ]

    be2 = [
        e for e in myG.myedges() if e.nodes[0].x > 98 and e.nodes[0].x < 104
        and e.nodes[0].y > 98 and e.nodes[0].y < 111
    ]

    be3 = [
        e for e in myG.myedges() if e.nodes[1].x > 98 and e.nodes[1].x < 104
        and e.nodes[1].y > 98 and e.nodes[1].y < 111
    ]

    return be + be2 + be3


def define_capetown_barriers(myG):

    be = [e for e in myG.myedges() if e.nodes[0].x < 136 and e.nodes[0].x > 15]
    be2 = [e for e in be if e.nodes[1].x < 136 and e.nodes[1].x > 15]

    be3 = [e for e in be2 if e.nodes[0].y < 12 and e.nodes[1].y < 12]
    todrop = [
        e for e in be3 if e.nodes[0].x > 15 and e.nodes[0].x < 65
        and e.nodes[0].y > 6 and e.nodes[1].y > 6
    ]

    for e in be3:
        if abs(e.rads) > math.pi / 4:
            todrop.append(e)

    be4 = [e for e in be3 if e not in todrop]

    return be4


if __name__ == "__main__":

    # filename = "data/CapeTown"
    # place = "cape"

    filename = "data/epworth_demo"
    place = "ep"

    original = mgh.import_and_setup(filename,
                                    threshold=1,
                                    byblock=False,
                                    name=place)
    original.define_roads()
    original.define_interior_parcels()

    # barriers = define_capetown_barriers(original)
    barriers = define_epworth_barriers(original)
    mgh.build_barriers(barriers)

    original.plot_roads()
    master = original.copy()

    mgh.build_all_roads(original, wholepath=True, barriers=True)

    original.plot_roads(master=master)

    plt.show()