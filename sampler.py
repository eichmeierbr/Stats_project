#!/usr/bin/env python

import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.cluster import KMeans
from parameter import *

from pyDOE import lhs

class Sampler:
    def __init__(self, method = 'lhc', params=2, samples=9):
        self.num_params = params
        self.num_samples = samples
        self.method = method
        self.params = []

    def getSamplesLHS(self, criterion='m'): # Other options = c, cm, corr
        print("lhs")
        return lhs(self.num_params, self.num_samples, criterion=criterion)

    def getSamplesRandom(self):
        print("random")
        return np.random.uniform(0,1, (self.num_samples, self.num_params))

    def getSamplesGrid(self):
        print("grid")
        n_pts = self.num_samples
        n_dims = self.num_params
        sim_pts = 7000

        ls = []
        for i in range(n_dims):
            ls.append(np.linspace(0,1,int(np.round(np.power(sim_pts,1/float(n_dims)),0))))
        sim_pts = list(itertools.product(*ls))


        l = KMeans(n_pts).fit(sim_pts).cluster_centers_
        return l


    def getRawSamples(self):
        if self.method == 'lhs':
            return self.getSamplesLHS()
        elif self.method == 'grid':
            return self.getSamplesGrid()
        else:
            return self.getSamplesRandom()

    def getSamples(self, params, numSamples=5, numParams=9, useParams=True):
        self.num_params = len(params)
        self.params = params
        self.num_samples = numSamples

        rawSamples = self.getRawSamples()

        if not useParams: 
            return rawSamples
            
        outSamples = []
        for i in range(self.num_samples):
            thisSet = []
            for j in range(self.num_params):
                thisSet.append( params[j].convertValueToParameter(rawSamples[i,j]) )
            outSamples.append(thisSet)

        return outSamples


if __name__ == "__main__":
    param = Parameter([0,1], 0.5)

    methods = ['random','lhs','grid']
    lhc = Sampler()
    for meth in methods:
        lhc.method = meth
        lhd = lhc.getRawSamples()
        xs = lhd[:,0]
        ys = lhd[:,1]

        plt.title(lhc.method)
        plt.scatter(xs,ys)
        plt.show()
