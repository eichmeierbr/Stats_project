#!/usr/bin/env python
import numpy as np

class Parameter:
    def __init__(self, values, count=0, lin=True, categ=False):
        self.options = values
        self.value = values[0]
        self.linear = lin # Linear vs. Lograithmic scaling
        self.samples = 0
        if count==0:
            self.samples = len(values)
        else:
            self.samples = count
        self.categorical = categ

  
    def convertValueToParameter(self, sample):
        if self.categorical:
            sample *= self.samples
            return self.options[int(sample)]

        if self.linear:
            minVal = self.options[0]
            maxVal = self.options[1]
            val = minVal + sample * (maxVal-minVal)
            return val
        else:
            minVal = np.log(self.options[0])
            maxVal = np.log(self.options[1])
            val = np.exp(minVal + sample * (maxVal-minVal))
            return val