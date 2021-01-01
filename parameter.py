#!/usr/bin/env python
import numpy as np

class Parameter:
    def __init__(self, values, count=0, lin=True, categ=False, name=None):
        self.options = values
        self.value = values[0]
        self.linear = lin # Linear vs. Lograithmic scaling
        self.samples = 0
        self.name = name
        if count==0:
            self.samples = len(values)
        else:
            self.samples = count
        self.categorical = categ

  
    def convertValueToParameter(self, sample):
        sample = max(sample, 1e-6)
        sample = min(sample, 1)

        if self.categorical:
            sample *= self.samples
            val = self.options[int(sample)]

        elif self.linear:
            minVal = self.options[0]
            maxVal = self.options[1]
            val = minVal + sample * (maxVal-minVal)
        else:
            minVal = np.log(self.options[0])
            maxVal = np.log(self.options[1])
            val = np.exp(minVal + sample * (maxVal-minVal))

        return val

    def setValueFromSample(self,sample):
        self.value = self.convertValueToParameter(sample)