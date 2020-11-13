#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy import stats
from sampler import *


def voronoi_volumes(points, v):
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return vol


if __name__ == "__main__":

    methods = ['random','lhs','grid']
    lhc = Sampler()
    for meth in methods:
        lhc.method = meth
        lhd = lhc.getRawSamples()
        xs = lhd[:,0]
        ys = lhd[:,1]

        dists = []
        for i in range(len(lhd)):
            for j in range(i+1,len(lhd)):
                dists.append(np.linalg.norm(lhd[i]-lhd[j]))
        vals = stats.kstest(lhd[:,0], 'norm')
        # vals = stats.chisquare(lhd[:,0])

        vor = Voronoi(lhd)
        vols = voronoi_volumes(lhd,vor)
       
        print(lhc.method)
        print(vals)
        print('Median Dist: %f' %(np.median(dists)))
        print('Mean Dist: %f' %(np.mean(dists)))
        print('Dev Dist: %f' %(np.std(dists)))
        # plt.scatter(xs, ys)
        fig = voronoi_plot_2d(vor)
        plt.title(lhc.method)
        plt.show()