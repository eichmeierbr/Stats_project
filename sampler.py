#!/usr/bin/env python

import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from parameter import *

from pyDOE import lhs

class Sampler:
    def __init__(self, params=2, samples=9):
        self.num_params = params
        self.num_samples = samples
        self.method = 'lhs'
        self.params = []

    def getSamplesLHS(self, criterion='m'): # Other options = c, cm, corr
        return lhs(self.num_params, self.num_samples, criterion=criterion)

    def getSamplesRandom(self):
        return np.random.uniform(0,1, (self.num_samples, self.num_params))

    def getSamplesGrid(self):
        p_copy = self.params[:]
        ls = []
        if not p_copy:
            for i in range(self.num_params):
                ls.append(np.linspace(0,1,int(np.sqrt(self.num_samples))))
        else:
            for i in range(len(p_copy)):
                ls.append(np.linspace(0,1,p_copy[i].samples))
        l = list(itertools.product(*ls))
        for i in range(len(l)):
            l[i] = list(l[i])
        return np.array(l)


    def getRawSamples(self):
        if self.method == 'lhs':
            return self.getSamplesLHS()
        elif self.method == 'grid':
            return self.getSamplesGrid()
        else:
            return self.getSamplesRandom()

    def getSamples(self, params, numSamples=5, numParams=9, method='lhs', useParams=True):
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