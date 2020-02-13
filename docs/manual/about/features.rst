Features
=================

Statistical Tools
-----------------

"Optimization":  Given any objective function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima:


"Sampling": Given any probability distribution of type $f:\mathbb{R}^N\rightarrow\mathbb{R}$, Korali can draw samples to determine its distribution: 
  
    $$\vartheta^{(k)} \sim f.$$

"Bayesian Inference":    Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:

    $$ p(\vartheta | d)  =  \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

    Allowing users to compute the maximum a posteriori estimate:

    $$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  p(\vartheta | d) $$

    Or sample its posterior distribution:

    $$\vartheta^{(k)} \sim p(\vartheta | d).$$
  
Hierarchical Bayesian Modeling:   Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:

    $$ p(\vartheta | d)  =  \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

    Allowing users to compute the maximum a posteriori estimate:

    $$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  p(\vartheta | d) $$
  
    Or sample its posterior distribution:

    $$\vartheta^{(k)} \sim p(\vartheta | d).$$

Design Principles
-------------------

Scalable Performance:  The Korali work distribution engine has been optimized to fully harness computational resources of large-scale supercomputers, maximizing throughput and minimizing workload imbalance.   Furthermore, Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.
  
Intuitive Interface:  Korali provides an fully descriptive language-independent interface that requires only modest programming knowledge.

Extensibile Engine:  Korali is designed as a completely modular and extensible software. Researchers can easily integrate and test new statistical methods for optimization and sampling into Korali. Likewise, new problems types can be easily added into Korali.

Multi-Language: Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Fortran/Python and pre-compiled computational models.

