# Korali  ![](images/templogo.png){: style="height:25px;width:25px"}

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## What Korali does for you

### [Optimization](/tutorials/optimization/optimization)
Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,$

### [Sampling](/tutorials/sampling/sampling)
Treat $f$ as an unnormalized density and draw samples to determine its distribution: $\vartheta^{(k)} \sim f.$  

### [Bayesian Inference](/tutorials/bayesian/bayesian)

Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:
$$
p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\,
$$

Allowing users to compute the maximum a posteriori estimate: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, p(\theta | d) \,,$ or sample its posterior distribution: $\vartheta^{(k)} \sim p(\theta | d).$

## Design Principles

### [Extensibility](usage/basics)

Korali is designed as a completely modular and extensible software. Operations can be changed on runtime to produce multiple results without the need of re-compilation.

### [Multi-Language](usage/basics)

Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Python and pre-compiled computational models.

### [Multi-Intrusiveness](usage/config/#evaluation-conduit)

A novelty in Korali is its support for multi-intrusive evaluation. That is, Korali can sample from computational models either semi-instrusively, through static linking or dynamic loading of Python modules; or nonintrusively, through shell command execution.

### [High-Throughput](usage/conduits/distributed.md)

Korali's work distribution engine has been optimized to fully harness computational resources, maximizing throughput and minimizing latency and load imbalance.

### [High-Performance](usage/conduits/distributed.md)

Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.

