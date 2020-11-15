import numpy
from ga import*
from ga_car_pole import*


######### test initial population here #########
# num_generations = 10
# population_size = 10
# num_params = 10
# num_parents = 4
# num_mutations = 1
# GA_agent = GA(population_size,num_params,num_parents,num_mutations)

# best_outputs = []
# for generation in range(num_generations):
#     best_loss, best_gene = GA_agent.get_next_gen(generation) 
#     best_outputs.append(best_loss)




##########Test GA cartpole
range_low = 0.00022
range_high = 0.00025
num_generations = 10
population_size = 10
num_params = 1
num_parents = 4
num_mutations = 1
GA_agent = GA_CAR_POLE(population_size,num_params,num_parents,num_mutations,range_low, range_high)

best_outputs = []
for generation in range(num_generations):
    best_loss, best_gene = GA_agent.get_next_gen(generation) 
    print("Genration", generation)
    print("best loss", best_loss)
    print("best gene", best_gene)
    best_outputs.append(best_loss)



import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("loss")
matplotlib.pyplot.show()