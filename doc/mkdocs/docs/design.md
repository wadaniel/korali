# Design

## Principles

We have designed Korali based on these key principles:

### [Ease of Use](usage/basics)

Korali provides an intuitive language-independent interface to designing optimization and uncertainty quantification problems.

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

## References

### Papers & Posters

+ [(μ,λ)-CCMA-ES for Constrained Optimization with an Application in Pharmacodynamics](https://doi.org/10.1145/3324989.3325725), 
*G. Arampatzis*, *D. Wälchli*, P. Weber, H. Rästas, *P. Koumoutsakos*

+ [The CMA Evolution Stragegy: A Tutorial](https://github.com/AlexanderFabisch/CMA-ESpp), N. Hansen

+ [Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)](https://www.mitpressjournals.org/doi/10.1162/106365603321828970), N. Hansen, S. Mueller, *P. Koumoutsakos*

+ [Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging](https://ascelibrary.org/doi/full/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29),
by J. Ching and Y. Chen

### Lectures & Presentations

+ Sergio M. Martin, [**Lecture: The Korali Framework**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_Lecture_Korali.pdf.pdf) <br>
ETH Zürich, Spring 2019

+ Sergio M. Martin, [**Lecture: High-Throughput Sampling**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/04/hpcse2-19_Lecture_UPC.pdf) <br>
ETH Zürich, Spring 2019 

+ Georgios Arampatzis, [**Lecture: Optimization using Covariance Matrix Adaptation (CMA)**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_cmaes.pdf) <br>
ETH Zürich, Spring 2019 

+ Georgios Arampatzis, [**Lecture: Transitional Markov Chain Monte Carlo (TMCMC)**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_tmcmc.pdf) <br>
ETH Zürich, Spring 2019 

