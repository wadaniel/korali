**************************
Deep Learner
**************************

Uses a deep neural networks (DNN) to solve a supervised learning problem, in which training/validation sets are provided.

It employs three DNNs:

- Training Network - Used to adjust the weights and biases of the neural layers with the help of a user-defined optimizer. The objective function for the optimizer is given by a user-definable loss function (e.g., Mean Square Error), based on the inferred values and actual solution. 
- Validation Network - Used to measure the improvement (in terms of loss function values) of the training network with regards to inferring results.  
- Test Network - The resulting DNN that the solver uses to infer new values for input test sets provided via the test() function.

