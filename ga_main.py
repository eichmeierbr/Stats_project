import numpy
import copy
from ga import*
from loss_functions import*


######### test initial population here #########
# param = Parameter([0,100])
# params = []

# num_generations = 100
# population_size = 10
# for i in range(population_size): params.append(copy.copy(param))
# num_parents = 4
# num_mutations = 1
# GA_agent = GA(population_size,params,num_parents,num_mutations, test_loss, approx_rate=0, method='lhc')
# best = GA_agent.Big_Funct(num_generations, show_stats=True)


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
lr_param = Parameter([1e-4, 1], lin=False)
maxGradNormParam = Parameter([0, 1])
params = [lr_param, maxGradNormParam]

num_generations = 20
population_size = 10
num_parents = 4
num_mutations = 1
GA_agent = GA(population_size,params,num_parents,num_mutations, cartPoleLoss, approx_rate=0, method='grid')
best = GA_agent.Big_Funct(num_generations, show_stats=True)