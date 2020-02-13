***************************
Introduction to Korali
***************************

Phasellus tincidunt enim id gravida pulvinar. Morbi sollicitudin turpis quis magna ornare facilisis. Cras mollis lacus odio, in posuere risus posuere vitae. Cras eleifend est sit amet malesuada porttitor. Aliquam erat volutpat. Aenean nec faucibus purus, bibendum bibendum arcu. Nunc ac euismod lacus. Fusce pellentesque lorem odio, ut scelerisque leo rhoncus nec. Fusce et semper nisl. Nulla a gravida sem, at molestie nunc. Cras quis ligula ut justo malesuada convallis nec in nisl.

Donec pulvinar enim id ligula molestie, at ultrices est tempor. Ut ornare, quam ut ultricies sollicitudin, enim risus elementum est, eget cursus nisl ipsum vitae mauris. Morbi vestibulum augue risus. Morbi orci orci, dictum interdum ligula ut, vehicula feugiat velit. Morbi tincidunt, tellus vitae vehicula luctus, metus turpis finibus metus, pulvinar pharetra leo tortor interdum mi. Vestibulum pretium ante dolor, at bibendum mi volutpat at. Pellentesque massa magna, rhoncus non urna vel, porttitor posuere erat. Cras sit amet lobortis erat, eu tristique lorem. Phasellus posuere magna non varius porta. Nunc consequat ultrices massa, sit amet pellentesque augue pharetra ac. Ut posuere tincidunt magna. Donec ornare elit ante, quis malesuada risus pulvinar et. Quisque viverra gravida ex, sed efficitur enim fermentum id. Phasellus ut augue et dolor iaculis molestie.

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


Publications
========================

Korali Team
=========================

Korali Team
------------

Sergio
George
Daniel
petros

Collaborators
--------------

Contributing

Affiliations
--------------
