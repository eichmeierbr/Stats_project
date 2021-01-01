import numpy
import copy
from ga import*
from loss_functions import*
from sklearn.neural_network import MLPClassifier,MLPRegressor

######### test initial population here #########
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
best = GA_agent.Big_Funct(num_generations, show_stats=True)


##########Test GA cartpole
# range_low = 0
# range_high = 1
# num_generations = 10
# population_size = 10
# num_params = 4
# num_parents = 4
# num_mutations = 1
# GA_agent = GA_CAR_POLE(population_size,num_params,num_parents,num_mutations,range_low, range_high)

# best_outputs = []
# for generation in range(num_generations):
#     best_loss, best_gene = GA_agent.get_next_gen(generation) 
#     print("Genration", generation)
#     print("best loss", best_loss)
#     print("best gene", best_gene)
#     best_outputs.append(best_loss)




##########Test GA cartpole
lr_param = Parameter([0.000001, 1])
maxGradNormParam = Parameter([0.000001, 1])
params = [lr_param, maxGradNormParam]

num_generations = 12
population_size = 8
num_parents = 2
num_mutations = 1

clf = MLPRegressor(hidden_layer_sizes=(100,100),activation='relu',tol=.1, alpha=0.2,
                        solver='lbfgs', learning_rate='adaptive', verbose=0,  random_state=9)

GA_agent = GA(population_size,params,num_parents,num_mutations, cartPoleLoss, approx_rate=1, method='grid',clf=clf)
best = GA_agent.Big_Funct(num_generations, show_stats=True)





############ Test NN ############
# act_functs = ['relu', 'tanh']
# sizes = [25,50, 100]
# num_layers = [1,2,3]

# act_param = Parameter(act_functs, categ=True, name='Activation Functions')
# nodes_param = Parameter(sizes, categ=True, name='Nodes Per Layer')
# layer_param = Parameter(num_layers, categ=True, name='Number of Layers')
# momentum_param = Parameter([0,1], name='Momentum')
# params = [act_param, nodes_param, layer_param, momentum_param]


# num_generations = 20
# population_size = 10
# num_parents = 4
# num_mutations = 1
# GA_agent = GA(population_size,params,num_parents,num_mutations, neuralNetLoss, approx_rate=0, method='lhs')
# best = GA_agent.Big_Funct(num_generations, show_stats=True)
