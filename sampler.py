#!/usr/bin/env python

import copy
import numpy as np
import matplotlib.pyplot as plt
import itertools
from parameter import *

from pyDOE import lhs

class Sampler:
    def __init__(self):
        self.num_params = 2
        self.num_samples = 9
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

    def getSamples(self, params, numSamples=5, method='lhs'):
        self.num_params = len(params)
        self.params = params
        self.num_samples = numSamples

        rawSamples = self.getRawSamples()
        outSamples = []

        for i in range(self.num_samples):
            thisSet = []
            for j in range(self.num_params):
                thisSet.append( self.convertValueToParameter(params[j], rawSamples[i,j]) )
            outSamples.append(thisSet)

        return outSamples


    def convertValueToParameter(self, param, sample):
        if param.categorical:
            sample *= param.samples
            return param.options[int(sample)]

        if param.linear:
            minVal = param.options[0]
            maxVal = param.options[1]
            val = minVal + sample * (maxVal-minVal)
            return val
        else:
            minVal = np.log(param.options[0])
            maxVal = np.log(param.options[1])
            val = np.exp(minVal + sample * (maxVal-minVal))
            return val


if __name__ == "__main__":
    param = Parameter([0,1], 0.5)

    methods = ['random','lhs','grid']
    for meth in methods:
        lhc = Sampler()
        lhc.method = meth
        lhd = lhc.getRawSamples()
        xs = lhd[:,0]
        ys = lhd[:,1]
    
        plt.title(lhc.method)
        plt.scatter(xs, ys)
        plt.show()