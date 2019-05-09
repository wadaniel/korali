# ![](logos/tmp.svg){: style="height:25px;width:25px"} Korali

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## Design Principles

  - [**Modularity.**](usage/introduction.md) Korali is designed as a completely modular software. Each one of its modules can be replaced during runtime, without the need of re-compilation.
  - [**Scalability.**](usage/conduits/sequential.md) We have designed Korali's problem definition interface to remain agnostic about its execution platform (and even programming language). It follows that a Korali application running in a single-core will, if required to, also run efficiently in the thousands of cores of a supercomputer.
  - [**High-Throughput.**](usage/conduits/upcxx.md) Korali's has been optimized to fully utilize the given computational resources, maximizing sampling throughput while minimize sampling latency and communication overheads
  - [**High-Performance.**](usage/conduits/upcxx.md) Korali supports the execution of parallel (MPI, UPC++) and GPU-based (CUDA) computational models.

## What Korali does for you

  - [**Direct problems.**](tutorials/direct/direct.md) Given any function $f:\mathbb{R}\rightarrow\mathbb{R}^N$ Korali can
    - [**Optimize:**](tutorials/direct/direct.md#optimize) find the minima or maxima of $f$
    $$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
	$$

	- [**Sample:**](tutorials/direct/direct.md#sample) treat $f$ as an unnormalized density and draw samples from it
	$$
	\vartheta^{(k)} \sim f.
	$$  

  - [**Bayesian Inference.**](tutorials/bayesian/bayesian.md) Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build
the posterior distribution below and enabling the following operations:
    $$
	p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\, .
	$$

    - [**Compute the maximum a posteriori estimate**](tutorials/bayesian/bayesian.md#optimize)
    $$
    \vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, p(\theta | d) \,,
    $$

    - [**Sample the posterior distribution**](tutorials/bayesian/bayesian.md#sample)
    $$
	\vartheta^{(k)} \sim p(\theta | d).
    $$


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new features will be constantly added.
