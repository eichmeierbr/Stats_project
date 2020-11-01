#!/usr/bin/env python

from parameter import *
from hypercube import *


### Define Hyperparameter Options ###
lr          = [0.001, 1]
funct       = ['tanh', 'sigmoid', 'linear']

### Translate raw hyperparameters into Parameter class ###
lr_param        = Parameter(lr, 5, lin=False)
funct_param     = Parameter(funct,categ=True)
parameter_set   = [lr_param, funct_param]

### Create Sample set from parameters
lhc         = Hypercube()
ps          = lhc.getSamples(parameter_set)

print(ps)