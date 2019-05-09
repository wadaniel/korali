# ![](templogo.png){: style="height:25px;width:25px"} Korali

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## Design Principles

  - [**Modularity.**](usage/introduction.md) Korali is designed as a completely modular software. Each one of its modules can be replaced during runtime, without the need of re-compilation.
  - [**Scalability.**](usage/conduits/sequential.md) We have designed Korali's problem definition interface to remain agnostic about its execution platform (and even programming language). It follows that a Korali application running in a single-core will, if required to, also run efficiently in the thousands of cores of a supercomputer.
  - [**High-Throughput.**](usage/conduits/upcxx.md) Korali's has been optimized to fully utilize the given computational resources, maximizing sampling throughput while minimize sampling latency and communication overheads
  - [**High-Performance.**](usage/conduits/upcxx.md) Korali supports the execution of parallel (MPI, UPC++) and GPU-based (CUDA) computational models.

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


Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ korali will build
the posterior distribution
	$$
	p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\, .
	$$
Then, you have the options to:

  - [**compute the maximum a posteriori estimate**](quick_start/bayesian.md#optimize)
  $$
  \vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
  $$

  - [**sample the posterior distribution**](quick_start/bayesian.md#sample)
    $$
		\vartheta^{(k)} \sim p(\theta | d).
    $$

<br>

After [installing](installation.md) the software you can have a look at the [quick start](quick_start) guide or [examples](./examples/sampling.md) and learn how to run your own code.


# Korali Principles

(todo)

## Notes

Please send questions and feedback to:

- martiser AT ethz.ch
- garampat AT ethz.ch
- wadaniel AT ethz.ch


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new feature will be constantly added.
