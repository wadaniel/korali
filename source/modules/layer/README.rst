******
Layers
******

This module contains the descriptions of the layers to form the Neural Network (see modules/neuralNetwork). It is based on Intel's library oneDNN (https://github.com/oneapi-src/oneDNN).

Each layer conists of a vector :math:`\mathbf{z}^l\in\mathbb{R}^{n_l}`with *Node Count* elements. Furthermore each layers is connected to the previous layer by *Weights*, a *Bias* and an activation function *Activation Function*.

There is different types of layers:

Input layer
###########

Represents the input to the neural network, which has no previous layer.

Output layer
############

Represents the output of the neural network.

Dense layer
###########

Represent a hidden layers :math:`\mathbf{z}^l\in\mathbb{R}^{n_l}` with $l=2,\dots,L-1$ for a neural network with :math:`L` layers. The layer is fully-connected, meaning that every node of the layer are connected to every node of the previous and the next layer. In the forward operation the entries of the previous layer :math:`\mathbf{z}^{l-1}\in\mathbb{R}^{n_{l-1}}` are multiplied by a weight matrix :math:`W\in\mathbb{R}^{n_{l}\times n_{l-1}}` and a bias :math:`\mathbf{b}^l\in\mathbb{R}^{n_l}`. The resulting value for the hidden layer is computed by applying an activation function :math:`\varphi_l` element-wise

.. math::

    \mathbf{z}^l = \varphi_l(W^l \mathbf{z}^{l-1}+\mathbf{b}^l)

If we denote the components of the vectors by :math:`z_i^{l-1}` and :math:`z_j^{l}, b_j^l` and for the matrix by :math:`W_{ij}^{l}` for :math:`i=1,\dots,n_{l-1}` and :math:`j=1,\dots,n_{l}`, this operation can be written as

.. math::

    z_j^l = \varphi_l(W_{ij}^l z_i^{l-1}+b_j^l)