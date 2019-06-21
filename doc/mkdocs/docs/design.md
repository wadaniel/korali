# Design

## Principles

We have designed Korali based on these key principles:

### Ease of Use

Korali provides an intuitive JSON-based interface that is mostly language-independent.

### Extensibility

Korali is designed as a completely modular and extensible software. Researchers can easily integrate and test new statistical methods for optimization and sampling into Korali. Likewise, new problems (e.g., sensor placement) can be easily added into Korali's engine.

### Multi-Language

Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Fortran/Python and pre-compiled computational models.

### Multi-Intrusiveness

A novelty in Korali is its support for multi-intrusive evaluation. That is, Korali can sample from computational models either semi-instrusively, through static linking or dynamic loading of Python modules; or nonintrusively, through shell command execution.

### High-Throughput

Korali's work distribution engine has been optimized to fully harness computational resources of large-scale supercomputers, maximizing throughput and minimizing latency and load imbalance.

### High-Performance

Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.

## References

### Papers & Posters

+ [(μ,λ)-CCMA-ES for Constrained Optimization with an Application in Pharmacodynamics](https://doi.org/10.1145/3324989.3325725), by
*G. Arampatzis*, *D. Wälchli*, P. Weber, H. Rästas, *P. Koumoutsakos*

+ [The CMA Evolution Stragegy: A Tutorial](https://github.com/AlexanderFabisch/CMA-ESpp), by N. Hansen

+ [Reducing the Time Complexity of the Derandomized Evolution Strategy with Covariance Matrix Adaptation (CMA-ES)](https://www.mitpressjournals.org/doi/10.1162/106365603321828970), by N. Hansen, S. Mueller, *P. Koumoutsakos*

+ [Transitional Markov Chain Monte Carlo Method for Bayesian Model Updating, Model Class Selection, and Model Averaging](https://ascelibrary.org/doi/full/10.1061/%28ASCE%290733-9399%282007%29133%3A7%28816%29),
by J. Ching and Y. Chen

+ [Self-Adapting Control Parameters in Differential Evolution: A Comparative Study on Numerical Benchmark Problems](https://ieeexplore.ieee.org/document/4016057), by Brest et. al.

+ [DRAM: Efficient adaptive MCMC](https://link.springer.com/article/10.1007%2Fs11222-006-9438-0), by Heikki Haario

### Lectures & Presentations

+ Sergio M. Martin, [**Lecture: The Korali Framework**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_Lecture_Korali.pdf.pdf) <br>
ETH Zürich, Spring 2019

+ Sergio M. Martin, [**Lecture: High-Throughput Sampling**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/04/hpcse2-19_Lecture_UPC.pdf) <br>
ETH Zürich, Spring 2019 

+ Georgios Arampatzis, [**Lecture: Optimization using Covariance Matrix Adaptation (CMA)**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_cmaes.pdf) <br>
ETH Zürich, Spring 2019 

+ Georgios Arampatzis, [**Lecture: Transitional Markov Chain Monte Carlo (TMCMC)**](https://www.cse-lab.ethz.ch/wp-content/uploads/2019/03/hpcse2-19_tmcmc.pdf) <br>
ETH Zürich, Spring 2019 

