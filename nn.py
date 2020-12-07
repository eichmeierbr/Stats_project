import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier,MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from sampler import *

matplotlib.use('TkAgg') 
# matplotlib.use('Qt5Agg')

def Plot_stuff(x1,y1,x2,y2):


    ax2 = plt.subplot(121, projection='3d')
    ax2.scatter(x1[:,0],x1[:,1],y1)
    ax2.set_title("training data")

    ax3 = plt.subplot(122, projection='3d')
    ax3.scatter(x2[:,0],x2[:,1],y2)
    ax3.set_title("output")
 




    plt.show()

    return
lr_param = Parameter([0.0, .2])
maxGradNormParam = Parameter([0, 1])
params = [lr_param, maxGradNormParam]
number_o_samples = 300 
temp = Sampler('grid',len(params),number_o_samples)
X_test = np.array(temp.getSamples(params,numSamples=number_o_samples))


X_train = np.loadtxt("Xvals.txt")
Y_train = np.loadtxt("Yvals.txt")
clf = MLPRegressor(hidden_layer_sizes=(100,100),activation='relu', max_iter=3*len(Y_train), alpha=0.001,
                     solver='lbfgs', learning_rate='adaptive', verbose=1,  random_state=9)

clf.fit(X_train.astype('float'), Y_train.astype('float64').ravel())
Y_test = clf.predict(X_test.astype('float'))

# clf = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu', max_iter=1, alpha=20,
#                      solver='lbfgs', learning_rate='adaptive', verbose=1,  random_state=9)

# clf.fit(X_train.astype('int'), Y_train.astype('int').ravel())
# Y_test = clf.predict(X_test.astype('int'))



Y_test = np.array(Y_test)



# x = np.arange(0, 5, 0.25)
# y = np.arange(0, 5, 0.25)
# x, y = np.meshgrid(x, y)
# r = np.sqrt(x**2 + y**2)
# z = np.sin(r)

Plot_stuff(X_train,Y_train,X_test,Y_test)


print("done")