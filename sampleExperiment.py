#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# # from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
# # from scipy import stats
# # from sklearn.cluster import KMeans
# import itertools
from sampler import *
from ga import*
from loss_functions import*
import seaborn as sns


def getSampleLoss(params, lossFunc):
    vals = np.random.uniform(0,1,size=len(params))
    for i in range(len(vals)):
        params[i].setValueFromSample(vals[i])
    loss = lossFunc(params)
    return loss



if __name__ == "__main__":

    # methods = ['random','lhs','grid']
    # methodsDict = {'random':'Random', 'lhs':'Latin Hypercube','grid':'Uniform'}
    # samps = [5,12,25]
    # n_pts =10
    # n_dims = 2
    # sim_pts = 12000
    # lhc = Sampler(params=n_dims, samples=n_pts)

    # fig, axs = plt.subplots(3,3)
    # fontweight = 22
    # plt.rcParams.update({'font.size': fontweight})

    # for i in range(len(samps)):
    #     num = samps[i]
    #     for j in range(len(methods)):
    #         plt.rcParams.update({'font.size': 22})

    #         meth = methods[j]
    #         lhc.num_samples = num
    #         lhc.method = meth
    #         lhd = lhc.getRawSamples()
    #         xs = lhd[:,0]
    #         ys = lhd[:,1]
    
    #         if i==0:
    #             # axs[i,j].set_title('Num Samples %i' %(num))
    #             axs[i,j].set_title(methodsDict[meth])
    #         if j==0:
    #             axs[i,j].set_ylabel('%i' %(num), fontsize=22)
    #         axs[i,j].set_xticks([])
    #         axs[i,j].set_xticklabels([])
    #         axs[i,j].set_yticks([])
    #         axs[i,j].set_yticklabels([])
    #         axs[i,j].scatter(xs, ys)


    # plt.rcParams.update({'font.weight': 'bold'})
    # fig.suptitle('Sampling Comparison', fontweight='bold')
    # fig.text(0.04, 0.5, 'Number of Samples', va='center', rotation='vertical')
    # plt.show()


    # ls = []
    # for i in range(n_dims):
    #     ls.append(np.linspace(0,1,int(np.round(np.power(sim_pts,1/float(n_dims)),0))))
    # sim_pts = list(itertools.product(*ls))


    # samples = KMeans(n_pts).fit(sim_pts).cluster_centers_

    # if n_dims == 2:
    #     plt.scatter(samples[:,0], samples[:,1])

    # elif n_dims == 3:
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.scatter(samples[:,0], samples[:,1], samples[:,2])
    # plt.show()




    # a = [[[0.1378427521303115, 492.0], [0.26800311907726604, 492.0], [0.5837932661603281, 492.0], [0.7402337246921062, 492.0], [0.8920911240889483, 492.0], [0.1378427521303115, 463.0], [0.26800311907726604, 463.0], [0.5837932661603281, 463.0], [0.7402337246921062, 463.0], [0.8920911240889483, 463.0], [0.1378427521303115, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.8920911240889483, 491.0], [0.1378427521303115, 490.0], [0.26800311907726604, 490.0], [0.5837932661603281, 490.0], [0.7402337246921062, 490.0], [0.8920911240889483, 490.0], [0.1378427521303115, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.8920911240889483, 491.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 490.0], [0.7402337246921062, 490.0], [0.5837932661603281, 490.0], [0.8920911240889483, 490.0], [0.4071553901088778, 490.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.7402337246921062, 491.0], [0.5837932661603281, 491.0], [0.8920911240889483, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 490.0], [0.7402337246921062, 490.0], [0.5837932661603281, 490.0], [0.8920911240889483, 490.0], [0.4071553901088778, 490.0], [0.7402337246921062, 492.0], [0.4071553901088778, 492.0], [0.26800311907726604, 492.0], [0.5837932661603281, 492.0], [0.7402337246921062, 492.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.7402337246921062, 490.0], [0.4071553901088778, 490.0], [0.26800311907726604, 490.0], [0.5837932661603281, 490.0], [0.7402337246921062, 490.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0], [0.7402337246921062, 491.0], [0.4071553901088778, 491.0], [0.26800311907726604, 491.0], [0.5837932661603281, 491.0], [0.7402337246921062, 491.0]], [[0.39510565143184745, 492.0], [0.583797065921324, 492.0], [0.325857272497648, 492.0], [0.836132755543006, 492.0], [0.40039543367236397, 492.0], [0.39510565143184745, 463.0], [0.583797065921324, 463.0], [0.325857272497648, 463.0], [0.836132755543006, 463.0], [0.40039543367236397, 463.0], [0.39510565143184745, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.836132755543006, 491.0], [0.40039543367236397, 491.0], [0.39510565143184745, 490.0], [0.583797065921324, 490.0], [0.325857272497648, 490.0], [0.836132755543006, 490.0], [0.40039543367236397, 490.0], [0.39510565143184745, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.836132755543006, 491.0], [0.40039543367236397, 491.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 490.0], [0.836132755543006, 490.0], [0.325857272497648, 490.0], [0.40039543367236397, 490.0], [0.836132755543006, 490.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.836132755543006, 491.0], [0.325857272497648, 491.0], [0.40039543367236397, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 490.0], [0.836132755543006, 490.0], [0.325857272497648, 490.0], [0.40039543367236397, 490.0], [0.836132755543006, 490.0], [0.836132755543006, 492.0], [0.836132755543006, 492.0], [0.583797065921324, 492.0], [0.325857272497648, 492.0], [0.8915274015913549, 492.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0], [0.836132755543006, 490.0], [0.836132755543006, 490.0], [0.583797065921324, 490.0], [0.325857272497648, 490.0], [0.8915274015913549, 490.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0], [0.836132755543006, 491.0], [0.836132755543006, 491.0], [0.583797065921324, 491.0], [0.325857272497648, 491.0], [0.8915274015913549, 491.0]]]
    # a = np.array(a)
    # for i in range(len(a)):
    #     a[i] = np.sort(a[i])
    # # plt.plot(a[0,:,0], a[0,:,1])
    # # plt.show()

    # for i in range(len(a[0,0])):
    #     fig, ax = plt.subplots()
    #     # ax.set_aspect("equal")
    #     hist, xbins, ybins, im = ax.hist2d(a[0,:,0], a[0,:,1], bins=10)
    #     for i in range(len(ybins)-1):
    #         for j in range(len(xbins)-1):
    #             ax.text(xbins[j]+0.04, ybins[i]+1, hist.T[i,j], 
    #                     color="w", ha="center", va="center", fontweight="bold")
    #     plt.show()

    # plt.scatter(a[0,:,0], a[0,:,1])
    # plt.xlabel('Parameter Value')
    # plt.ylabel('Loss')
    # plt.show()
    # b = 3


# methods = ['random','lhs','grid']
# methodsDict = {'random':'Random', 'lhs':'Latin Hypercube','grid':'Uniform'}
# samps = [5,12,25]
# n_pts =10
# n_dims = 2
# sim_pts = 12000
# lhc = Sampler(params=n_dims, samples=n_pts)

# fig, axs = plt.subplots(3,3)
# fontweight = 22
# plt.rcParams.update({'font.size': fontweight})

# for i in range(len(samps)):
#     num = samps[i]
#     for j in range(len(methods)):
#         plt.rcParams.update({'font.size': 22})

#         meth = methods[j]
#         lhc.num_samples = num
#         lhc.method = meth
#         lhd = lhc.getRawSamples()
#         xs = lhd[:,0]
#         ys = lhd[:,1]

#         if i==0:
#             # axs[i,j].set_title('Num Samples %i' %(num))
#             axs[i,j].set_title(methodsDict[meth])
#         if j==0:
#             axs[i,j].set_ylabel('%i' %(num), fontsize=22)
#         axs[i,j].set_xticks([])
#         axs[i,j].set_xticklabels([])
#         axs[i,j].set_yticks([])
#         axs[i,j].set_yticklabels([])
#         axs[i,j].scatter(xs, ys)


# plt.rcParams.update({'font.weight': 'bold'})
# fig.suptitle('Sampling Comparison', fontweight='bold')
# fig.text(0.04, 0.5, 'Number of Samples', va='center', rotation='vertical')
# plt.show()


    lossFunc = test_loss
    vals = [1,10,20,30,40,50,60,70,80,90]
    # param = Parameter(vals, categ=True)
    param = Parameter([0, 100])
    params = []
    lossThresh = 10

    num_generations = 100
    population_size = 10
    for i in range(population_size): 
        param.name= str(vals[i])
        params.append(copy.copy(param))
    num_parents = 4
    num_mutations = 1


    all_vals = {}
    iters = 100
    methodsDict = {'random':'Random', 'lhs':'Latin Hypercube', 'grid':'Uniform','ran':'Full Random'}
    # methodsDict = {'ran':'Full Random'}
    for meth, v in methodsDict.items():

        vals = []
        if meth == 'ran':
            pass
        #     for i in range(iters):
        #         for j in range(1000):
        #             loss = getSampleLoss(params,lossFunc)
        #             if loss < 40: print(loss)
        #             if loss < lossThresh:
        #                 vals.append(j)
        #                 break
        else:
            GA_agent = GA(population_size,params,num_parents,num_mutations, lossFunc, approx_rate=0, method=meth)
            for i in range(iters):
                print('Evaluating Iter: %i/%i' %(i,iters))
                GA_agent.reset()
                best, losses = GA_agent.Big_Funct(num_generations, show_stats=False, return_losses=True)
                idx = -1
                for gen in losses:
                    idx +=1
                    if gen[1] < lossThresh: 
                        break
                vals.append(idx)

        all_vals[v] = vals
    ax = sns.displot(all_vals, kind="kde", clip=[0,num_generations*population_size])
    plt.xlabel('Iterations Until Convergence')
    plt.title('Sampling Success Distributions')
    plt.show()