import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier,MLPRegressor
from mpl_toolkits.mplot3d import Axes3D
from sampler import *

matplotlib.use('TkAgg') 
# matplotlib.use('Qt5Agg')

def Plot_stuff(gx,gy,x1,y1,x2,y2):

    ax1 = plt.subplot(131, projection='3d')
    ax1.scatter(gx[:,0],gx[:,1],gy)
    ax1.set_title("actual data")

    ax2 = plt.subplot(132, projection='3d')
    ax2.scatter(x1[:,0],x1[:,1],y1)
    ax2.set_title("training data")

    ax3 = plt.subplot(133, projection='3d')
    ax3.scatter(x2[:,0],x2[:,1],y2)
    ax3.set_title("output")
 




    plt.show()

    return

num_params = 2
decimals = 0
data_points = 10
big_points = 500

def function_val(val):
    stuff = min(val[0],4)**2+min(val[1],4)**2+np.random.rand(1)*10-5
    return stuff

X_good = []
for i in range(big_points):
    X_good.append(list(np.random.rand(2)*10-5))
Y_good=[]
for i in range(big_points):
    val = X_good[i]
    Y_good.append(function_val(val))

X_train = []
for i in range(data_points):
    X_train.append(list(np.random.rand(2)*10-5))
Y_train=[]
for i in range(data_points):
    val = X_train[i]
    Y_train.append(function_val(val))
###############
X_train = []
temp = Sampler('grid', 2, data_points)
X_train = np.array(temp.getRawSamples())*10-5
Y_train=[]
for i in range(data_points):
    val = X_train[i]
    Y_train.append(function_val(val))
###############
X_test = []
for i in range(big_points):
    X_test.append(list(np.random.rand(2)*10-5))

X_good = np.array(X_good)
Y_good = np.array(Y_good)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

clf = MLPRegressor(hidden_layer_sizes=(100,100),activation='relu', max_iter=5, alpha=0.001,
                     solver='sgd', learning_rate='adaptive', verbose=1,  random_state=9)

clf.fit(X_train.astype('float'), Y_train.astype('float64').ravel())
Y_test = clf.predict(X_test.astype('float'))



Y_test = np.array(Y_test)*10**decimals



# x = np.arange(0, 5, 0.25)
# y = np.arange(0, 5, 0.25)
# x, y = np.meshgrid(x, y)
# r = np.sqrt(x**2 + y**2)
# z = np.sin(r)

Plot_stuff(X_good,Y_good,X_train,Y_train,X_test,Y_test)


print("done")