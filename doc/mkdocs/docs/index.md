# ![](images/templogo.png){: style="height:25px;width:25px"} Korali

**Korali** is a high-performance computing framework for optimization, sampling and Bayesian uncertainty quantification of large-scale computational models. [Here should be the abstract of our paper]

## What Korali does for you

### Optimization
  Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima:
    $$
	\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  \,\,\, f(\vartheta) \,,
	$$

### Sampling

  Treat $f$ as an unnormalized density and draw samples from it to learn its distribution:
	$$
	\vartheta^{(k)} \sim f.
	$$  

### Bayesian Inference 

Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$,  and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution below:
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

### Modularity

Korali is designed as a completely modular software. Each one of its modules can be replaced during runtime, without the need of re-compilation.

### Scalability

We have designed Korali's problem definition interface to remain agnostic about its execution platform (and even programming language). It follows that a Korali application running in a single-core will, if required to, also run efficiently in the thousands of cores of a supercomputer.

### High-Throughput

Korali's has been optimized to fully utilize the given computational resources, maximizing sampling throughput while minimize sampling latency and communication overheads

### High-Performance

Korali supports the execution of parallel (MPI, UPC++) and GPU-based (CUDA) computational models.

