import numpy
import copy
from ga import*
from loss_functions import*


######### test initial population here #########
def number_guess_test():
    vals = [1,10,20,30,40,50,60,70,80,90]
    # param = Parameter(vals, categ=True)
    param = Parameter([0, 100])
    params = []
    
    num_generations = 100
    population_size = 10
    for i in range(population_size): 
        param.name= str(vals[i])
        params.append(copy.copy(param))
    num_parents = 4
    num_mutations = 1
    GA_agent = GA(population_size,params,num_parents,num_mutations, test_loss, approx_rate=0, method='lhc')
    best = GA_agent.optimize_params(num_generations, show_stats=True)




##########Test GA cartpole
def test_cartpole():
    lr_param = Parameter([1e-4, 1], lin=False)
    maxGradNormParam = Parameter([0, 1])
    params = [lr_param, maxGradNormParam]
    # 
    num_generations = 20
    population_size = 10
    num_parents = 4
    num_mutations = 1
    GA_agent = GA(population_size,params,num_parents,num_mutations, cartPoleLoss, approx_rate=0, method='grid')
    best = GA_agent.optimize_params(num_generations, show_stats=True)





############ Test NN ############
def test_nn():
    act_functs = ['relu', 'tanh']
    sizes = [25,50, 100]
    num_layers = [1,2,3]
    # 
    act_param = Parameter(act_functs, categ=True, name='Activation Functions')
    nodes_param = Parameter(sizes, categ=True, name='Nodes Per Layer')
    layer_param = Parameter(num_layers, categ=True, name='Number of Layers')
    momentum_param = Parameter([0,1], name='Momentum')
    params = [act_param, nodes_param, layer_param, momentum_param]
    # 
    # 
    num_generations = 20
    population_size = 10
    num_parents = 4
    num_mutations = 1
    GA_agent = GA(population_size,params,num_parents,num_mutations, neuralNetLoss, approx_rate=0, method='lhs')
    best = GA_agent.optimize_params(num_generations, show_stats=True)


if __name__ == "__main__":
    # test_cartpole()
    # test_nn()
    number_guess_test()