**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## What Korali does for you

### **Optimization**
Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,$

### **Sampling**
Treat $f$ as an unnormalized density and draw samples to determine its distribution: $\vartheta^{(k)} \sim f.$  

### **Bayesian Inference**

Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:
$$
p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\,
$$

Allowing users to compute the maximum a posteriori estimate: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, p(\theta | d) \,,$ or sample its posterior distribution: $\vartheta^{(k)} \sim p(\theta | d).$

## Design Principles

### **Extensibility**

Korali is designed as a completely modular and extensible software. Operations can be changed on runtime to produce multiple results without the need of re-compilation.

### **Multi-Language**

Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Python and pre-compiled computational models.

### **Multi-Intrusiveness**

A novelty in Korali is its support for multi-intrusive evaluation. That is, Korali can sample from computational models either semi-instrusively, through static linking or dynamic loading of Python modules; or nonintrusively, through shell command execution.

### **High-Throughput**

Korali's work distribution engine has been optimized to fully harness computational resources, maximizing throughput and minimizing latency and load imbalance.

### **High-Performance**

Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.


## About us

The Korali Project is written and maintained by

* **Georgios Arampatzis**, garampat at ethz.ch
* **Sergio Miguel Martin**, martiser at ethz.ch
* **Daniel Waelchli**, wadaniel at ethz.ch
* **Petros Koumoutsakos**, petros at ethz.ch (Director)


## Credits

Korali's code includes a properly credited and licensed copy of [Niels Lohmann's Json for C++](https://github.com/nlohmann/json).
Our implementation of CMA-ES has been inspired by [Alexander Fabisch's](https://github.com/AlexanderFabisch/CMA-ESpp) C++ implementation.

