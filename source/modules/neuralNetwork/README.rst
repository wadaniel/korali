***************
Neural Network
***************

**For the time being, Korali Neural Networks can only use CMAES as optimizer. We are currently working on supporting other optimizers.**

This module combined the given :ref:`layers <module-layer>` to a Neural Network. The Neural Network can be used, for example, as a function approximator in a :ref:`Supervised Learning <module-problem-supervisedlearning>` problem. 

The current implementation is based on Intel's `oneDNN <https://github.com/oneapi-src/oneDNN>`_ library.

This module enables two *Operation* types:

 + **Training**: In this operation, the weights and biases are optimized to minimize a given loss function. The initial guess is chosen according to the specified *Weight Initialization*. 
 
 + **Inference**: In this operation, weights and biases are either provided by the user, or obtained by the training operation. This configuration is only used to compute the output for a given input.
 
