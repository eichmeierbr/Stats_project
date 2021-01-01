import numpy as np
from sampler import *
from parameter import *
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
matplotlib.use('TkAgg') 

# sns.displot([1,1], kind="kde")

def tempLearnerFunc(params):
    return 3

def tempTrainApproximator(X, Y):
    return

def Plot_stuff(x1,y1):


    ax2 = plt.subplot(121, projection='3d')
    ax2.scatter(x1[:,0],x1[:,1],y1)
    ax2.set_title("training data")


    plt.show()

    return

class GA(object):
    def __init__(self,population_size, parameters,num_parents,num_mutations, loss_func, approx_rate=0, method='grid',clf=None):
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
        self.want_approx = (approx_rate > 0) or (clf!=None)

        self.X_train=np.loadtxt("Xvals.txt")
        self.Y_train=np.loadtxt("Yvals.txt")
        self.clf=clf
        # self.clf.fit(np.array(self.X_train).astype('float'), np.array(self.Y_train).astype('float64').ravel())
        self.clf_loss=np.Inf
        

        ##Sampler would go here 
        self.range_low = 0
        self.range_high = 1    
        self.params = parameters
        self.method = method
        self.lhc = Sampler(method, self.num_params, population_size)

        # self.population = np.array(lhc.getSamples(self.params, population_size))
        self.population = np.array(lhc.getRawSamples())
        self.visited_points = self.population
        self.pop_hist = np.array([])

    def reset(self):
        self.loss_history = []
        self.population = np.array(self.lhc.getRawSamples())
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
                # lossFunc = tempLearnerFunc ########### Replace this as desired
                if self.clf!=None:
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


        # Y_predict = loss
        best_genes = []
        if approximatedLoss:
            # Setup loss storing or training
            # Y_predict = self.clf.predict(np.array(self.population).astype('float'))
            self.X_train = np.append(self.population,self.X_train,axis=0)
            self.Y_train = np.append(loss,self.Y_train,axis=0)
            self.clf.fit(np.array(self.X_train).astype('float'), np.array(self.Y_train).astype('float').ravel())
            if True:

                number_o_samples = 300
                temp = Sampler('lhs',len(self.params),number_o_samples)
                X_test = np.array(temp.getRawSamples())
                self.visited_points = np.append(self.visited_points,self.population,axis=0)
                X_test = np.append(X_test,self.visited_points,axis=0)
                Y_test = self.clf.predict(np.array(X_test).astype('float'))

                best_genes = []
                for i in range(self.num_parents):
                    val = np.argmin(Y_test)
                    best_genes.append(X_test[val,:])
                    Y_test[val]=np.Inf

                s = len(self.Y_train)-100
                # Plot_stuff(self.X_train[:s,:],self.Y_train[:s])


        return loss,np.array(best_genes)
    


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
                    random_val = np.random.uniform(self.range_low, offspring_crossover[offspring,random_index], 1)
                else:
                    random_val = np.random.uniform(offspring_crossover[offspring,random_index], self.range_high, 1)

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
        loss,clf_best_gene = self.calculate_loss()
        self.store_losses(loss)
    
        if mode == "deterministic":

            if generation>0:
                loss = np.append(loss,self.loss_o_parents,axis=0)
                self.population = np.append(self.population,self.parents,axis=0)

        self.best_loss, best_gene =  self.get_best_DNA(loss)
        self.parents , self.loss_o_parents = self.select_parents(loss)
        # self.clf_parent, self.loss_o_clf_parent = self.select_parents(Y_predict)
        if self.clf!=None:
            self.parents = np.append(self.parents,clf_best_gene,axis=0)
        # self.parents = clf_best_gene

        offspring_crossover = self.crossover(offspring_size=(self.pop_size[0]-self.parents.shape[0], self.num_params))
        self.offspring_mutation = self.mutation(offspring_crossover)
        
        
        if mode == 'deterministic':
            self.population = self.offspring_mutation
        else:

            self.population[0:self.parents.shape[0], :] = self.parents
            self.population[self.parents.shape[0]:, :] = self.offspring_mutation

        return self.best_loss, best_gene


    def store_losses(self, losses):
        for loss, sample in zip(losses, self.population):
            vals = []
            for i in range(len(self.params)):
                vals.append(self.params[i].convertValueToParameter(sample[i]))
            self.loss_history.append([vals, loss])


    def Big_Funct(self, num_generations=20, show_stats=False, return_losses=False):
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
        if return_losses:
            return self.params, self.loss_history
        else:
            return self.params


    def plotLossCurve(self, best_outputs):
        methodsDict = {'random':'Random', 'lhc':'Latin Hypercube','grid':'Uniform'}
        plt.plot(best_outputs)
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.title("Number Guessing (%s Sampling)" %(methodsDict[self.method]))
        plt.show()        


    def plotTrainingHistograms(self):
        # Plot Histograms
        a = self.loss_history[:]
        thresh = 100
        for j in reversed(range(len(a))):
            if a[j][1] > thresh:
                a.pop(j)

        for i in range(len(self.params)):
            param_vals = []
            for v in a:
                param_vals.append(v[0][i])
            is_log = not self.params[i].linear

            # fig, ax = plt.subplots()
            # hist, xbins, ybins, im = ax.hist2d(a[:,0,i], a[:,1,i], bins=10)
            # ax.set_xlabel('Parameter Value')
            # ax.set_ylabel('Loss')
            # plt.show()

            # plt.scatter(a[:,0,i], a[:,1,i])
            # plt.xlabel('Parameter Value')
            # plt.ylabel('Count')
 
            if self.params[i].categorical:
                ax = sns.countplot(x=param_vals)
                if not self.params[i].name==None:
                    plt.title('%s Distribution' %(self.params[i].name))
            else:    
                ax = sns.displot(param_vals, kind="kde", clip=self.params[i].options, log_scale=is_log)
                plt.xlabel('Values')
                if not self.params[i].name==None:
                    plt.title('%s Distribution' %(self.params[i].name))


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