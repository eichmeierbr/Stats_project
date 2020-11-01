#!/usr/bin/env python

import copy
import numpy as np
import matplotlib.pyplot as plt
from parameter import *

from pyDOE import lhs

class Hypercube:
    def __init__(self):
        self.num_params = 2
        self.num_samples = 10

    def getRawSamples(self, criterion='m'): # Other options = c, cm, corr
        return lhs(self.num_params, self.num_samples, criterion=criterion)

    def getSamples(self, params):
        self.num_params = len(params)
        self.num_samples = 0
        for p in params:
            self.num_samples = max(self.num_samples, p.samples)

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
    lhc = Hypercube()


    # lhd = lhs(2,10, criterion='m')
    lhd = lhc.getRawSamples()
    xs = lhd[:,0]
    ys = lhd[:,1]

    plt.scatter(xs, ys)
    plt.show()
    a =4