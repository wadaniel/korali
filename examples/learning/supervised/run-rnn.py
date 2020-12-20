#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import korali
k = korali.Engine()

# Setting random seed for reproducibility
np.random.seed(0xC0FFEE)

# Input parameters
Arch = "FNN"
#Arch = "RNN"
tf = 2.0 # Total Time
dt = 0.4 # Time Differential
B = 100 # Batch Size
s = 1.0  # Parameter for peak separation
w = np.pi # Parameter for wave speed
a = 1.0 # Scaling

# Transformation Function
def y(x, t): return np.sin(x * s +  w * t)  

X = np.random.uniform(0, np.pi*2, B)
T = np.arange(0, tf, dt)

trainingInputSet = [ ]
for i, t in enumerate(T):
 trainingInputSet.append([ ])
 for j, x in enumerate(X):
  trainingInputSet[i].append([x])
   
trainingSolutionSet = [ ] 
for i, t in enumerate(T):
 trainingSolutionSet.append([ ])
 for j, x in enumerate(X):
  trainingSolutionSet[i].append([y(x,t)])

### Defining a learning problem to infer values of sin(x,t)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = len(T)
e["Problem"]["Training Batch Size"] = B
e["Problem"]["Inference Batch Size"] = B
e["Problem"]["Input"]["Data"] = trainingInputSet
e["Problem"]["Input"]["Size"] = 1
e["Problem"]["Solution"]["Data"] = trainingSolutionSet
e["Problem"]["Solution"]["Size"] = 1
 
### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 20
e["Solver"]["Optimizer"] = "AdaBelief"
e["Solver"]["Learning Rate"] = 0.0001

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "CuDNN"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 32

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Recurrent"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Output Channels"] = 32
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Mode"] = "GRU"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Recurrent"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 32
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Mode"] = "GRU"

### Configuring output 

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = 50
e["Random Seed"] = 0xC0FFEE
k.run(e)

### Obtaining inferred results from the NN and comparing them to the actual solution

testInputSet = [ ]
for i, t in enumerate(T):
 testInputSet.append([ ])
 for j, x in enumerate(X):
  testInputSet[i].append([x])
   
testSolutionSet = [ ] 
for i, t in enumerate(T):
 testSolutionSet.append([ ])
 for j, x in enumerate(X):
  testSolutionSet[i].append([y(x, t)])

testInferredSet = e.getEvaluation(testInputSet) 

cmap = cm.get_cmap(name='Set1')

xAxis = [ x[0] for x in testInputSet[0] ]
plt.plot(xAxis, testSolutionSet[-1], "o", color=cmap(i))
plt.plot(xAxis, testInferredSet[-1], "x", color=cmap(i))

### Calc MSE on test set

mse = np.mean((np.array(testInferredSet) - np.array(testSolutionSet))**2)
print("MSE on test set: {}".format(mse))

### Plotting Results

plt.show()
