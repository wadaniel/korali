***************
Neural Network
***************

This module combined the given *Layers* (see modules/layer) to a Neural Network. The Neural Network can be used as a function approximator in a supervised learning problem (see modules/problem/supervisedLearning). The operations are performed using Intel's library oneDNN (https://github.com/oneapi-src/oneDNN). Wheter the operation is performed on the CPU or on the GPU is specified by the *Engine Kind*.

The module allows two *Operation*s:

**Training** In this operation the weights and biases are optimized using the specified *Optimizer* (see modules/solver) to minimize the supervised learning objective. The initial guess is chosen according to the specified *Weight Initialization*. 

**Inference** In this operation weights and biases which are provided by the user, or obtained by the training operation are used to compute the output for a given input.
 