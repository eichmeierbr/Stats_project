import numpy
import ga



######### test initial population here #########
num_generations = 100
population_size = 10
num_parents = 4
num_params = 10
mutation = 5

range_low = 0
range_high = 100

pop_size = (population_size,num_params)
new_population = numpy.random.uniform(low=range_low, high=range_high, size=pop_size)
################################################



best_outputs = []
for generation in range(num_generations):
    
    loss = ga.calculate_loss(new_population)
    best_outputs.append(numpy.min(loss))
    print("Generation : ", generation+1)
    print("Best result : ", numpy.min(loss))
    
    parents = ga.select_parents(new_population, loss, num_parents)

    
    # print("loss: ", loss)
    
    # print("Parents: ", parents)


    offspring_crossover = ga.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0], num_params))
    offspring_mutation = ga.mutation(offspring_crossover, range_low, range_high, num_mutations=mutation)


    new_population[0:parents.shape[0], :] = parents
    new_population[parents.shape[0]:, :] = offspring_mutation

    print(" ")
    



import matplotlib.pyplot
matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("loss")
matplotlib.pyplot.show()