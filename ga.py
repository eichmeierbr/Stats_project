import numpy as np

def calculate_loss(population):
    # this is for my simple tests, definitly needs to be improved
    arr = np.tile(np.array([1,10,20,30,40,50,60,70,80,90]), (population.shape[0], 1))
    val = population-arr
    loss = np.linalg.norm(val,axis=1)
    return loss

def select_parents(population, loss, num_parents):
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(loss == np.min(loss))[0][0]
        parents[i, :] = population[max_fitness_idx, :]
        # np.delete(population,max_fitness_idx)
        loss[max_fitness_idx] = np.Inf
    return parents

def crossover(parents, offspring_size):
    #####TODO: crossover with only two parents, need to be changeable
    offspring = np.empty(offspring_size)
    ############################################
    crossover_point = int(offspring_size[1]/2) # point at which crossover takes place between two parents
    # crossover_point = random #TODO
    ############################################

    for i in range(offspring_size[0]):
        # print(i)
        parent1_idx = i%parents.shape[0]
        parent2_idx = (i+1)%parents.shape[0]

        offspring[i, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[i, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover,range_low,range_high, num_mutations=1):
    for offspring in range(offspring_crossover.shape[0]):
        for i in range(num_mutations):
            random_index = np.random.randint(0, offspring_crossover.shape[1], 1)
            random_val = np.random.uniform(range_low, range_high, 1)
            offspring_crossover[offspring,random_index] = random_val
        # offspring_crossover
    return offspring_crossover
    # return offspring_crossover