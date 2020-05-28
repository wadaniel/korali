**************************
Neural Network
**************************

Represents a deep neural network comprised of an input layer, k hidden layers, and an output layer.

It accepts a Korali optimizer as parameter to optimize the network's weights and biases. It performs normalization on input and output batch.

On "Training" operation, it allows the calculation of gradients on backward propagation. On "Inference", it can only be forward-propagated (this is a faster operation since no auxiliary structures are saved for the calculation of the gradients).
 