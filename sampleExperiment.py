#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy import stats
from sklearn.cluster import KMeans
from sampler import *
import itertools


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
    n_pts =10
    n_dims = 2
    sim_pts = 7000
    lhc = Sampler(params=n_dims, samples=n_pts)

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


    # ls = []
    # for i in range(n_dims):
    #     ls.append(np.linspace(0,1,int(np.round(np.power(sim_pts,1/float(n_dims)),0))))
    # sim_pts = list(itertools.product(*ls))


    # samples = KMeans(n_pts).fit(sim_pts).cluster_centers_

    # if n_dims == 2:
    #     plt.scatter(samples[:,0], samples[:,1])

    # elif n_dims == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(samples[:,0], samples[:,1], samples[:,2])
    # plt.show()
