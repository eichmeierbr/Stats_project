import numpy
from ga import*
from ga_car_pole import*


######### test initial population here #########
range_low = 0
range_high = 100
num_generations = 100
population_size = 10
num_params = 10
num_parents = 4
num_mutations = 1
GA_agent = GA(population_size,num_params,num_parents,num_mutations,range_low,range_high)

best_outputs = []
for generation in range(num_generations):
    best_loss, best_gene = GA_agent.get_next_gen(generation) 
    best_outputs.append(best_loss)




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
range_low = 0.00001
range_high = 1
num_generations = 20
population_size = 10
num_params = 2
num_parents = 4
num_mutations = 1
GA_agent = GA_CAR_POLE(population_size,num_params,num_parents,num_mutations,range_low, range_high)

best_outputs = []
for generation in range(num_generations):
    best_loss, best_gene = GA_agent.get_next_gen(generation) 
    print("Generation", generation)
    print("best loss", best_loss)
    print("best gene", best_gene)
    best_outputs.append(best_loss)



import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Generation")
matplotlib.pyplot.ylabel("Loss")
matplotlib.pyplot.title("Cart Pole (random sampling)")
matplotlib.pyplot.show()