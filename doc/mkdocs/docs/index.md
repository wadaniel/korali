# ![](templogo.png){: style="height:25px;width:25px"} Korali

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## Design Principles

1. [**Modularity.**](usage/introduction.md) Korali is designed as a completely modular software. Each one of its modules can be replaced during runtime, without the need of re-compilation.
2. [**Scalability.**](usage/conduits/sequential.md) We have designed Korali's problem definition interface to remain agnostic about its execution platform (and even programming language). It follows that a Korali application running in a single-core will, if required to, also run efficiently in the thousands of cores of a supercomputer.
3. [**High-Throughput.**](usage/conduits/upcxx.md) Korali's has been optimized to fully utilize the given computational resources, maximizing sampling throughput while minimize sampling latency and communication overheads
4. [**High-Performance.**](usage/conduits/upcxx.md) Korali supports the execution of parallel (MPI, UPC++) and GPU-based (CUDA) computational models.


## What Korali does for you

### [1. Direct problems](tutorials/direct.md)

Given any function $f:\mathbb{R}\rightarrow\mathbb{R}^N$ korali can

  - [**Optimize:**](tutorials/direct.md#optimize) find the minima or maxima of $f$
    $$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
	$$

  - [**Sample:**](tutorials/direct.md#sample) treat $f$ as an unnormalized density and draw samples from it
  $$
  \vartheta^{(k)} \sim f.
  $$  



### [2. Bayesian Inference](tutorials/bayesian.md)


2. [**Uncertainty Quantification**](tutorials/bayesian.md): given a set of data $d$, respectively the output of a model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali samples the posterior distribution
	$$
	p(\vartheta | d) = \frac{p(d | \vartheta) \, p(\vartheta)}{p(d)}\, .
	$$
    I.e. the output is a set of samples $\theta^{(k)}$ with $k \in \{1,...,N\}$ such that
    $$
		\vartheta^{(k)} \sim p(\theta | d).
    $$
    Where $x$ are the input parameters of the computational model $f$.

<br>


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new feature will be constantly added.
