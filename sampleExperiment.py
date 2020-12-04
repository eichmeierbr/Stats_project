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
    methodsDict = {'random':'Random', 'lhs':'Latin Hypercube','grid':'Uniform'}
    samps = [5,12,25]
    n_pts =10
    n_dims = 2
    sim_pts = 12000
    lhc = Sampler(params=n_dims, samples=n_pts)

    fig, axs = plt.subplots(3,3)
    fontweight = 22
    plt.rcParams.update({'font.size': fontweight})

    for i in range(len(samps)):
        num = samps[i]
        for j in range(len(methods)):
            plt.rcParams.update({'font.size': 22})

            meth = methods[j]
            lhc.num_samples = num
            lhc.method = meth
            lhd = lhc.getRawSamples()
            xs = lhd[:,0]
            ys = lhd[:,1]
    
            if i==0:
                # axs[i,j].set_title('Num Samples %i' %(num))
                axs[i,j].set_title(methodsDict[meth])
            if j==0:
                axs[i,j].set_ylabel('%i' %(num), fontsize=22)
            axs[i,j].set_xticks([])
            axs[i,j].set_xticklabels([])
            axs[i,j].set_yticks([])
            axs[i,j].set_yticklabels([])
            axs[i,j].scatter(xs, ys)


    plt.rcParams.update({'font.weight': 'bold'})
    fig.suptitle('Sampling Comparison', fontweight='bold')
    fig.text(0.04, 0.5, 'Number of Samples', va='center', rotation='vertical')
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




    # a = [[[0.1378427521303115, 492.0], [0.26800311907726604, 492.0], [0.5837932661603281, 492.0], [0.7402337246921062, 492.0], [0.8920911240889483, 492.0], [0.1378427521303115, 463.0], [0.26800311907726604, 463.0], [0.5837932661603281, 463.0], [0.7402337246921062, 463.0], [0.8920911240889483, 463.0], [0.1378427521303115, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.8920911240889483, 491.0], [0.1378427521303115, 490.0], [0.26800311907726604, 490.0], [0.5837932661603281, 490.0], [0.7402337246921062, 490.0], [0.8920911240889483, 490.0], [0.1378427521303115, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.8920911240889483, 491.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 490.0], [0.7402337246921062, 490.0], [0.5837932661603281, 490.0], [0.8920911240889483, 490.0], [0.4071553901088778, 490.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 490.0], [0.7402337246921062, 490.0], [0.5837932661603281, 490.0], [0.8920911240889483, 490.0], [0.4071553901088778, 490.0], [0.7402337246921062, 492.0], [0.4071553901088778, 492.0], [0.26800311907726604, 492.0], [0.5837932661603281, 492.0], [0.7402337246921062, 492.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.7402337246921062, 490.0], [0.4071553901088778, 490.0], [0.26800311907726604, 490.0], [0.5837932661603281, 490.0], [0.7402337246921062, 490.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0]], [[0.39510565143184745, 492.0], [0.583797065921324, 492.0], [0.325857272497648, 492.0], [0.836132755543006, 492.0], [0.40039543367236397, 492.0], [0.39510565143184745, 463.0], [0.583797065921324, 463.0], [0.325857272497648, 463.0], [0.836132755543006, 463.0], [0.40039543367236397, 463.0], [0.39510565143184745, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.836132755543006, 491.0], [0.40039543367236397, 491.0], [0.39510565143184745, 490.0], [0.583797065921324, 490.0], [0.325857272497648, 490.0], [0.836132755543006, 490.0], [0.40039543367236397, 490.0], [0.39510565143184745, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.836132755543006, 491.0], [0.40039543367236397, 491.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 490.0], [0.836132755543006, 490.0], [0.325857272497648, 490.0], [0.40039543367236397, 490.0], [0.836132755543006, 490.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 490.0], [0.836132755543006, 490.0], [0.325857272497648, 490.0], [0.40039543367236397, 490.0], [0.836132755543006, 490.0], [0.836132755543006, 492.0], [0.836132755543006, 492.0], [0.583797065921324, 492.0], [0.325857272497648, 492.0], [0.8915274015913549, 492.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0], [0.836132755543006, 490.0], [0.836132755543006, 490.0], [0.583797065921324, 490.0], [0.325857272497648, 490.0], [0.8915274015913549, 490.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0]]]
    # a = np.array(a)
    # for i in range(len(a)):
    #     a[i] = np.sort(a[i])
    # # plt.plot(a[0,:,0], a[0,:,1])
    # # plt.show()

    # for i in range(len(a[0,0])):
    #     fig, ax = plt.subplots()
    #     # ax.set_aspect("equal")
    #     hist, xbins, ybins, im = ax.hist2d(a[0,:,0], a[0,:,1], bins=10)
    #     for i in range(len(ybins)-1):
    #         for j in range(len(xbins)-1):
    #             ax.text(xbins[j]+0.04, ybins[i]+1, hist.T[i,j], 
    #                     color="w", ha="center", va="center", fontweight="bold")
    #     plt.show()

    # plt.scatter(a[0,:,0], a[0,:,1])
    # plt.xlabel('Parameter Value')
    # plt.ylabel('Loss')
    # plt.show()
    # b = 3


methods = ['random','lhs','grid']
methodsDict = {'random':'Random', 'lhs':'Latin Hypercube','grid':'Uniform'}
samps = [5,12,25]
n_pts =10
n_dims = 2
sim_pts = 12000
lhc = Sampler(params=n_dims, samples=n_pts)

fig, axs = plt.subplots(3,3)
fontweight = 22
plt.rcParams.update({'font.size': fontweight})

for i in range(len(samps)):
    num = samps[i]
    for j in range(len(methods)):
        plt.rcParams.update({'font.size': 22})

        meth = methods[j]
        lhc.num_samples = num
        lhc.method = meth
        lhd = lhc.getRawSamples()
        xs = lhd[:,0]
        ys = lhd[:,1]

        if i==0:
            # axs[i,j].set_title('Num Samples %i' %(num))
            axs[i,j].set_title(methodsDict[meth])
        if j==0:
            axs[i,j].set_ylabel('%i' %(num), fontsize=22)
        axs[i,j].set_xticks([])
        axs[i,j].set_xticklabels([])
        axs[i,j].set_yticks([])
        axs[i,j].set_yticklabels([])
        axs[i,j].scatter(xs, ys)


plt.rcParams.update({'font.weight': 'bold'})
fig.suptitle('Sampling Comparison', fontweight='bold')
fig.text(0.04, 0.5, 'Number of Samples', va='center', rotation='vertical')
plt.show()
