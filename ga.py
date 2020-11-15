import numpy as np
from sampler import *
from parameter import *


class GA(object):
    def __init__(self,population_size,num_params,num_parents,num_mutations,range_low, range_high):
        self.pop_size = (population_size,num_params)
        self.num_params = num_params
        self.num_parents = num_parents
        self.num_mutations = num_mutations
        self.parents = None
        self.offspring_mutation = None


        ##Sampler would go here 
        self.range_low = range_low
        self.range_high = range_high    
        params = []
        for i in range(self.num_params):
            params.append(Parameter([self.range_low, self.range_high]))
        lhc = Sampler('lhs',num_params, population_size)

        self.population = np.array(lhc.getSamples(params, population_size, method='random'))

    ##THIS function will interact will get loss from leaner
    def calculate_loss(self):
        # this is for my simple tests, definitly needs to be improved
        arr = np.tile(np.array([1,10,20,30,40,50,60,70,80,90]), (self.population.shape[0], 1))
        val = self.population-arr
        loss = np.linalg.norm(val,axis=1)
        return loss

    def select_parents(self, loss):
        parents = np.empty((self.num_parents, self.population.shape[1]))
        for i in range(self.num_parents):
            max_fitness_idx = np.where(loss == np.min(loss))[0][0]
            parents[i, :] = self.population[max_fitness_idx, :]
            # np.delete(population,max_fitness_idx)
            loss[max_fitness_idx] = np.Inf
        self.parents = parents

    def crossover(self, offspring_size):
        #####TODO: crossover with only two parents, need to be changeable
        offspring = np.empty(offspring_size)
        ############################################
        crossover_point = int(offspring_size[1]/2) # point at which crossover takes place between two parents
        # crossover_point = random #TODO
        ############################################

        for i in range(offspring_size[0]):
            # print(i)
            parent1_idx = i%self.parents.shape[0]
            parent2_idx = (i+1)%self.parents.shape[0]

            offspring[i, 0:crossover_point] = self.parents[parent1_idx, 0:crossover_point]
            offspring[i, crossover_point:] = self.parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover):
        for offspring in range(offspring_crossover.shape[0]):
            for i in range(self.num_mutations):
                random_index = np.random.randint(0, offspring_crossover.shape[1], 1)
                random_val = np.random.uniform(self.range_low, self.range_high, 1)
                offspring_crossover[offspring,random_index] = random_val
        return offspring_crossover

    def get_best_DNA(self,loss):
        best_loss = np.min(loss)
        best_gene_idx = np.argmin(loss)
        best_gene = self.population[best_gene_idx,:]
        return best_loss, best_gene

    def get_next_gen(self,generation):
        if generation>0:
            self.population[0:self.parents.shape[0], :] = self.parents
            self.population[self.parents.shape[0]:, :] = self.offspring_mutation
        loss = self.calculate_loss()
        self.select_parents(loss)
        offspring_crossover = self.crossover(offspring_size=(self.pop_size[0]-self.parents.shape[0], self.num_params))
        self.offspring_mutation = self.mutation(offspring_crossover)

        best_loss, best_gene =  self.get_best_DNA(loss)
        return best_loss, best_gene