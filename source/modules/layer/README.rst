******
Layers
******

This module contains the descriptions of the layers to form a :ref:`Neural Network <module-neuralnetwork>`. 

Its implementation is based on Intel's `oneDNN <https://github.com/oneapi-src/oneDNN>`_ library.

Each layer consists of a vector :math:`\mathbf{z}^l\in\mathbb{R}^{n_l}` with *Node Count* elements. Each layer is connected to the previous layer by *Weights*, a *Bias* and an activation function *Activation Function*.

Korali exposes the following types of layers:

 + *Input layer*: Represents the input to the neural network, which has no previous layer.
 
 + *Output layer*: Represents the output of the neural network.
 
 + *Dense layers*: Represent fully-connected, feed-forward hidden layers of the neural network.
