#!/usr/bin/env python

from parameter import *
from sampler import *


### Define Hyperparameter Options ###
lr          = [0.001, 1]
funct       = ['tanh', 'sigmoid', 'linear']
numLayers       = [1, 2, 3, 4]
nodesPerLayer   = [10,20,50,100,200]
convNet         = [False, True]

### Translate raw hyperparameters into Parameter class ###
lr_param        = Parameter(lr, 5, lin=False)
funct_param     = Parameter(funct,categ=True)
layers_param    = Parameter(numLayers, categ=True)
nodes_param     = Parameter(nodesPerLayer, categ=True)
conv_param      = Parameter(convNet, categ=True)
parameter_set   = [lr_param, funct_param, layers_param, nodes_param, conv_param]

### Create Sample set from parameters
lhc         = Sampler()
ps          = lhc.getSamples(parameter_set, 10)

print(ps)