# Hyperparameter Tuning with a Genetic Algorithm

This project uses a genetic algorithm to effeciently select hyperparameters. Details of the algorithm and expected results can be found in the [Presentation](./documentation/hyperparameterTuning_FinalPresentation.pdf) and [Report](documentation/hyperparameterTuning_FinalReport.pdf). The package was developed for the course Statisical Techniques in Robotics at Carnegie Mellon University with Shaun Ryer and Stefan Zhu. 

## Usage

There are two primary files the user must modify to add a custom environment:

### loss_functions.py

This file holds the loss functions, or task environments, for the optimizer. A loss function can be defined anywhere so long as it is visible to ga_main.py.

A loss function must have the following criteria to succesfully run:
1. The parameters are input as a vector of the parameter class type. The input array represents a single parameter set.
2. The function must return a single loss value for the input parameter set. High performing parameter sets should receive low values for loss. For example, a classifier that correctly classifies 97% of test data could return a loss of 0.03.

### ga_main.py

This file acts as the overall runner for the optimizer. There are several examples in the file to help you implement your own environment. There are only a few steps to get your code running:

1. Create an array of parameters to optimize. The parameters can be categorical or continuous. If you select categorical, the 'values' argument should include all of the possible options and set categ=True. If your values are continuous, 'values' reflects the high and low bounds for optimization. See [parameter.py](./parameter.py) for details. For reporting purposes you can also name the parameters.
   
2. The next step is to initialize the GA. Several examples are given, but you can see the constructor in [ga.py](ga.py).
   
3. Call GA_agent.optimize_params() to begin the optimization process.


## Notes

In the report, we detail a method to efficiently search the estimated hyperparameter->loss mapping using a neural network. This concept worked well in testing, however it has not been merged into master. It can be found in the ml-approximation branch.

## Dependencies

* matplotlib
* numpy
* stable-baselines - for RL environment only
* openAI zoo - for RL environment only
* pyDOE
* seaborn