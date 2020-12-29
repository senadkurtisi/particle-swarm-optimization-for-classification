# particle-swarm-optimization-for-classification
Particle Swarm Optimization algorithm for detecting forged banknotes.

## The task
Particle Swarm Optimization algorithm was used in order to learn weights for multiplayer perceptron neural network for classification of banknotes.

## Dataset 
The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/banknote+authentication). 

## The algorithm
During execution classification error was used to estimate if one particle is better than the other. We try to minimze this error.
All of the algorithm hyperparameters can be easily changed. 

Hyperparameter values used to eachieve the results given below
| Swarm size  | Generations  | W  |  C1 | C2  | vmax |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 100  | 50  | 0.8 | 1.494  | 1.494  | 0.2  |

All of the weights were cliped to the value of **1.85**. This hyperparameter can be changed as well.


## The architecture
Neural network had 2 hidden layers with 5 and 4 neurons, respectively. Both of the hidden layers use ReLU activation, while the output layer (with just one unit) uses sigmoid activation. 

Number of neurons in each layer can be modified easily. Number of layers can be changed as well.

## Results
The algorithm was run 20 times independently.
Below you can see how the globally best solution (across all particles) changes for each of the runs. We can notice the decreasing trend for the classification error.

![Globally best for all 20 runs](20_runs_all_gens.png)

Below you can see the best solution found in each of these 20 runs.

![Globally best for all 20 runs](20_runs_final.png)

Below you can see the best solution per generation averaged across these 20 runs.

![Globally best for all 20 runs](20_runs_final.png)


### Evaluation on the test set
The solution which achieved lowest classification error on the training set was test on the test set, which is 20% of the entire dataset. The results are given below.

| Training acc  | Test acc  |
|:---:|:---:|
| 99.91 %  | 99.63 % |

As we can see the model generalizes well.


## Setup & instructions
**NOTE: This code was written in Python version 3.8.5**
**This doesn't mean that it won't work on the earlier version, it just means that it hasn't been tested.**

1. Open Anaconda prompt, or regular command prompt
2. Execute ```conda install requirements.txt``` or ```pip install requirements.txt``` (if you went for the second option with command prompt)
3. a) Run ```main.py``` to test the code from you IDE

   b) Execute: ```cd PATH_TO_THIS_REPO```. Execute ```python main.py```.
   
