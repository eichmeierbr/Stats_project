#dont run, already have 100 runs stored in Xvals.txt and Yvals.txt


from ga import*
from loss_functions import*
from sampler import *
import numpy as np

# param = Parameter([0,100])
# params = []
# numSamples = 3
# population_size = 10
# for i in range(population_size): params.append(copy.copy(param))

lr_param = Parameter([0.000001, 1])
maxGradNormParam = Parameter([0.000001, 1])
params = [lr_param, maxGradNormParam]

number_o_samples = 100
temp = Sampler('lhs',len(params),number_o_samples)
X_train = np.array(temp.getRawSamples())

y = []
print("")
for i in range(len(X_train)):
    print(i)
    for j in range(len(params)):
        params[j].setValueFromSample(X_train[i,j])
    y.append(cartPoleLoss(params))
    # y.append(test_loss(params))

print("")


np.savetxt("Xvals.txt",np.array(X_train))
np.savetxt("Yvals.txt",np.array(y))

