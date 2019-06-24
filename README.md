# Korali  ![](images/templogo.png){: style="height:25px;width:25px"}

**Korali** is a high-performance framework for Uncertainty Quantification of large-scale models.

### **What Korali Does**

??? "Function Optimization"
	Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}    f(\vartheta)$

??? "Function Sampling"
	Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can treat $f$ as an unnormalized density and draw samples to determine its distribution: $\vartheta^{(k)} \sim f.$

??? "Bayesian Inference"
	Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution: $$ p(\vartheta | d)  =  \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

	Allowing users to compute the maximum a posteriori estimate:

	$$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  p(\vartheta | d) $$

	Or sample its posterior distribution:

	$$\vartheta^{(k)} \sim p(\vartheta | d).$$

### **Design Principles**

??? "Ease of Use"
	Korali provides an intuitive JSON-based interface that is mostly language-independent.

??? "Extensibility"

	Korali is designed as a completely modular and extensible software. Researchers can easily integrate and test new statistical methods for optimization and sampling into Korali. Likewise, new problems types can be easily added into Korali's engine.

??? "Multi-Language"

	Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Fortran/Python and pre-compiled computational models.

??? "Multi-Intrusiveness"

	A novelty in Korali is its support for multi-intrusive evaluation. That is, Korali can sample from computational models either semi-instrusively, through static linking or dynamic loading of Python modules; or nonintrusively, through shell command execution.

??? "High-Performance"

	Korali's work distribution engine has been optimized to fully harness computational resources of large-scale supercomputers, maximizing throughput and minimizing latency and load imbalance.
	
	Furthermore, Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.