#!/usr/bin/env python3
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import korali
from mnist import MNIST

k = korali.Engine()

### Getting MNIST data
mndata = MNIST('./_data')
mndata.gz = True
images, labels = mndata.load_training()

### Converting images to Korali form (requires a time dimension)
imageVector = [ [ x ] for x in images ]

### Converting label data to (0,1) vector form
labelVector = [ ]
for l in labels:
 newLabel = np.zeros(10).tolist()
 newLabel[l] = 1
 labelVector.append(newLabel)
 
### Defining a learning problem to infer values of sin(x)

e = korali.Experiment()
e["Problem"]["Type"] = "Supervised Learning"
e["Problem"]["Max Timesteps"] = 1
e["Problem"]["Training Batch Size"] = len(images)
e["Problem"]["Inference Batch Size"] = 1

e["Problem"]["Input"]["Data"] = imageVector
e["Problem"]["Input"]["Size"] = len(images[0])
e["Problem"]["Solution"]["Data"] = labelVector
e["Problem"]["Solution"]["Size"] = 10

### Using a neural network solver (deep learning) for inference

e["Solver"]["Type"] = "Learner/DeepSupervisor"
e["Solver"]["Loss Function"] = "Mean Squared Error"
e["Solver"]["Steps Per Generation"] = 1
e["Solver"]["Learning Rate"] = 0.01

### Defining the shape of the neural network

e["Solver"]["Neural Network"]["Engine"] = "OneDNN"
e["Solver"]["Neural Network"]["Optimizer"] = "Adagrad"

e["Solver"]["Neural Network"]["Hidden Layers"][0]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][0]["Output Channels"] = 784

e["Solver"]["Neural Network"]["Hidden Layers"][1]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][1]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][2]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][2]["Output Channels"] = 800

e["Solver"]["Neural Network"]["Hidden Layers"][3]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][3]["Function"] = "Elementwise/Tanh"

e["Solver"]["Neural Network"]["Hidden Layers"][4]["Type"] = "Layer/Linear"
e["Solver"]["Neural Network"]["Hidden Layers"][4]["Output Channels"] = 10

e["Solver"]["Neural Network"]["Hidden Layers"][5]["Type"] = "Layer/Activation"
e["Solver"]["Neural Network"]["Hidden Layers"][5]["Function"] = "Softmax"

### Configuring output

e["Console Output"]["Frequency"] = 1
e["File Output"]["Enabled"] = False
e["Random Seed"] = 0xC0FFEE

### Training the neural network

e["Solver"]["Termination Criteria"]["Max Generations"] = 1000
k.run(e)

