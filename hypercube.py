#!/usr/bin/env python

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
  

param = Parameter([0,1], 0.5)
lhc = Hypercube()


# lhd = lhs(2,10, criterion='m')
lhd = lhc.getRawSamples()
xs = lhd[:,0]
ys = lhd[:,1]

plt.scatter(xs, ys)
plt.show()
a =4