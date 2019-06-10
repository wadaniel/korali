# ![](images/tmp.svg){: style="height:25px;width:25px"} Korali

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models.

## What Korali does for you

  - [**Optimization.**](tutorials/optimization/optimization.md) Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima:
    $$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
	$$

  - [**Sampling.**](tutorials/sampling/sampling.md) treat $f$ as an unnormalized density and draw samples from it:
	$$
	\vartheta^{(k)} \sim f.
	$$  

  - [**Bayesian Inference.**](tutorials/bayesian/bayesian.md) Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution below:
    $$
	p(\vartheta | d) \, = \, \frac{p(d | \vartheta) \,\, p(\vartheta)}{p(d)}\, .
	$$

	Allowing users to compute the maximum a posteriori estimate:
    $$
    \vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, p(\theta | d) \,,
    $$

    And sample its posterior distribution:
    $$
	\vartheta^{(k)} \sim p(\theta | d).
    $$


## Design Principles

  - [**Modularity.**](usage/introduction.md) Korali is designed as a completely modular software. Each one of its modules can be replaced during runtime, without the need of re-compilation.
  - [**Scalability.**](usage/conduits/sequential.md) We have designed Korali's problem definition interface to remain agnostic about its execution platform (and even programming language). It follows that a Korali application running in a single-core will, if required to, also run efficiently in the thousands of cores of a supercomputer.
  - [**High-Throughput.**](usage/conduits/upcxx.md) Korali's has been optimized to fully utilize the given computational resources, maximizing sampling throughput while minimize sampling latency and communication overheads
  - [**High-Performance.**](usage/conduits/upcxx.md) Korali supports the execution of parallel (MPI, UPC++) and GPU-based (CUDA) computational models.


<br><br><br><br><br><br>

!!! warning
    The software and the documentation page are under continuous development. New pages and new features will be constantly added.
