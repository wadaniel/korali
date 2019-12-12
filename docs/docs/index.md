# Korali  ![](images/logo.png){: style="height:25px;width:25px"}
<center>**High-performance framework for uncertainty quantification of large-scale models.**</center>

### **Download**

Latest Release: 13/12/2019 - [Korali v1.0.0](https://github.com/cselab/korali.git) - Build Status: [![CircleCI](https://circleci.com/gh/cselab/korali.svg?style=svg&circle-token=d73f56a4d14073880f8fe1140964afb58f2b1c35)](install/build)

 + [Installation Guide](install/install.md)
 + [Release Notes](https://github.com/cselab/korali/blob/master/docs/RELEASE-NOTES)
 + [System Requirements](install/#system-requirements)

### **Design Principles**

??? success "High-Performance"

  Korali's work distribution engine has been optimized to fully harness computational resources of large-scale supercomputers, maximizing throughput and minimizing latency and load imbalance.
  
  Furthermore, Korali supports the execution of parallel (OpenMP, Pthreads), distributed (MPI, UPC++), and GPU-based (CUDA) models.
  
??? success "Ease of Use"
  Korali provides an intuitive string-based interface that is mostly language-independent.

??? success "Extensibility"

  Korali is designed as a completely modular and extensible software. Researchers can easily integrate and test new statistical methods for optimization and sampling into Korali. Likewise, new problems types can be easily added into Korali's engine.

??? success "Multi-Language"

  Korali applications can be programmed in either C++ or Python. Additionally, Korali can sample from C++/Fortran/Python and pre-compiled computational models.

### **Latest News**

<a class="twitter-timeline" data-height="600" data-dnt="true" data-theme="light" href="https://twitter.com/EthKorali?ref_src=twsrc%5Etfw">Tweets by EthKorali</a> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>