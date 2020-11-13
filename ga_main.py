import numpy
from ga import*



######### test initial population here #########
num_generations = 1000
population_size = 10
num_params = 10
num_parents = 4
num_mutations = 1
GA_agent = GA(population_size,num_params,num_parents,num_mutations)

best_outputs = []
for generation in range(num_generations):
    best_loss, best_gene = GA_agent.get_next_gen(generation) 
    best_outputs.append(best_loss)


    


import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("loss")
matplotlib.pyplot.show()