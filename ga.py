import numpy as np
from sampler import *
from parameter import *

def tempLearnerFunc(params):
    return 3

def tempTrainApproximator(X, Y):
    return


class GA(object):
    def __init__(self,population_size, parameters,num_parents,num_mutations, loss_func, approx_rate=0, method='grid'):
        self.num_params = len(parameters)
        self.pop_size = (population_size,self.num_params)
        self.num_parents = num_parents
        self.num_mutations = num_mutations
        self.loss_history = []
        self.loss_function = loss_func
        # self.parents = []
        # self.offspring_mutation = None

        # Simulated Loss Function
        self.approx_rate = approx_rate
        self.approx_count = 0
        self.want_approx = approx_rate > 0

        ##Sampler would go here 
        self.range_low = 0
        self.range_high = 1    
        self.params = parameters
        lhc = Sampler(method, self.num_params, population_size)

        # self.population = np.array(lhc.getSamples(self.params, population_size))
        self.population = np.array(lhc.getRawSamples())
        self.pop_hist = np.array([])

    ##THIS function will interact will get loss from leaner
    def calculate_loss(self):
        loss = np.array([])

        # Select desired loss function
        lossFunc = self.loss_function
        approximatedLoss = False
        if self.want_approx:
            self.approx_count+=1
            if self.approx_count == self.approx_rate:
                self.approx_count = 0
                lossFunc = tempLearnerFunc ########### Replace this as desired
                approximatedLoss = True


        avg_num = 1
        for i in range(self.population.shape[0]):
            curr_loss = 0
            params = self.params[:]

            for j in range(len(params)):
                params[j].setValueFromSample(self.population[i,j])

            for k in range(avg_num):
                curr_loss += lossFunc(params)
            loss = np.append(loss, curr_loss/avg_num)

        if approximatedLoss:
            # Setup loss storing or training
            X = self.population
            Y = loss
            tempTrainApproximator(X, Y)

        return loss
    


    def select_parents(self, loss):
        parents = np.empty((self.num_parents, self.population.shape[1]))
        loss_list = []
        for i in range(self.num_parents):
            max_fitness_idx = np.where(loss == np.min(loss))[0][0]
            parents[i, :] = self.population[max_fitness_idx, :]
            # np.delete(population,max_fitness_idx)
            loss_list.append(loss[max_fitness_idx])
            loss[max_fitness_idx] = np.Inf
        return parents,loss_list

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
                rand = np.random.uniform(0, 1, 1)
                if rand>.5:
                    random_val = np.random.uniform(self.range_low, self.parents[0][random_index], 1)
                else:
                    random_val = np.random.uniform(self.parents[0][random_index], self.range_high, 1)

                # random_val = max(0,min(np.random.normal(self.parents[random_index], self.range_high*.5, 1),1))

                # random_val = np.max(random_val,self.range_low)
                # random_val = np.min(random_val,self.range_high)

                offspring_crossover[offspring,random_index] = random_val
        return offspring_crossover

    def get_best_DNA(self,loss):
        best_loss = np.min(loss)
        best_gene_idx = np.argmin(loss)
        best_gene = self.population[best_gene_idx,:]
        return best_loss, best_gene

    def get_next_gen(self,generation, mode = "non-deterministic"):
        # print(self.parents)
        # print(self.population)
        

        # if generation>0:
        loss = self.calculate_loss()
        self.store_losses(loss)
    
        if mode == "deterministic":

            if generation>0:
                loss = np.append(loss,self.loss_o_parents,axis=0)
                self.population = np.append(self.population,self.parents,axis=0)

        self.best_loss, best_gene =  self.get_best_DNA(loss)
        self.parents , self.loss_o_parents = self.select_parents(loss)
        offspring_crossover = self.crossover(offspring_size=(self.pop_size[0]-self.parents.shape[0], self.num_params))
        self.offspring_mutation = self.mutation(offspring_crossover)
        
        
        if mode == 'deterministic':
            self.population = self.offspring_mutation
        else:

            self.population[0:self.parents.shape[0], :] = self.parents
            self.population[self.parents.shape[0]:, :] = self.offspring_mutation

        return self.best_loss, best_gene


    def store_losses(self, losses):
        for loss in losses:
            for sample in self.population:
                if len(self.loss_history) == 0: 
                    for val in sample:
                        self.loss_history.append([[val, loss]])
                else:
                    for i in range(len(sample)):
                        self.loss_history[i].append([sample[i], loss])


    def Big_Funct(self, num_generations=20, show_stats=False):
        best_outputs = []
        self.loss_history = []
        for generation in range(num_generations):
            # if show_stats:
                # self.plotPopulation()
            best_loss, best_gene = self.get_next_gen(generation)
            best_gene_params = []
            for i in range(len(best_gene)):
                self.params[i].setValueFromSample(best_gene[i])
                best_gene_params.append(self.params[i].value)

            if show_stats:
                print("Generation", generation)
                print("best loss", best_loss)
                print("best gene", best_gene_params)
                best_outputs.append(best_loss)

        if show_stats:
            self.plotLossCurve(best_outputs)
            self.plotTrainingHistograms()
        return self.params


    def plotLossCurve(self, best_outputs):
        plt.plot(best_outputs)
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.title("Number Guessing (LHC Sampling)")
        plt.show()        


    def plotTrainingHistograms(self):
        # Plot Histograms
        a = np.array(self.loss_history)
        for i in range(len(a)):
            for j in range(len(a[0])):
                a[i,j] = self.params[i].convertValueToParameter(a[i,j])

        # for i in range(len(a)):
            # a[i] = np.sort(a[i])

        for i in range(len(a)):
            fig, ax = plt.subplots()
            # ax.set_aspect("equal")
            hist, xbins, ybins, im = ax.hist2d(a[i,:,0], a[i,:,1], bins=10)
            # for k in range(len(ybins)-1):
            #     for j in range(len(xbins)-1):
                    # ax.text(xbins[j]+0.04, ybins[k]+1, hist.T[k,j], 
                            # color="w", ha="center", va="center", fontweight="bold")
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Loss')
            plt.show()

            plt.scatter(a[i,:,0], a[i,:,1])
            plt.xlabel('Parameter Value')
            plt.ylabel('Loss')

            plt.show()

    def plotPopulation(self):
        if len(self.pop_hist) == 0:
            self.pop_hist = self.population
        else:
            self.pop_hist = np.vstack((self.pop_hist, self.population))
        plt.clf()
        plt.scatter(self.pop_hist[:,0], self.pop_hist[:,1])
        plt.ion()
        plt.show(block=False)
        plt.pause(0.01)