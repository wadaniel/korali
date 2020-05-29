************
Deep Learner
************

Uses a *Neural Networks* (see modules/neuralNetwork) to solve a supervised learning problem (see modules/problems/supervisedLearning).

It employs three Neural Networks:

- *Training Network* - Used to adjust the weights and biases with the help of a user-defined optimizer to minimize an objective function given in the supervised learning problem (e.g., Mean Square Error).
- *Validation Network* - Used to measure the improvement (in terms of loss function values) of the training network on the validation data given in the supervised learning problem.  
- *Test Network* - The result of the optimisation procedure which can be used to evaluate the neural network on a test set using the test() function.