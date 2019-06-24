# Korali  ![](images/templogo.png){: style="height:25px;width:25px"}

**Korali** is a high-performance framework for Uncertainty Quantification of large-scale models.

### **What Korali does for you**

??? "Function Optimization"
	Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima: $\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta} ,,, f(\vartheta) ,,$

??? "Function Sampling"
	Given any function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can treat $f$ as an unnormalized density and draw samples to determine its distribution: $\vartheta^{(k)} \sim f.$

??? "Bayesian Inference"
	Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution: $$ p(\vartheta | d) , = , \frac{p(d | \vartheta) ,, p(\vartheta)}{p(d)}, $$

	Allowing users to compute the maximum a posteriori estimate:

	$$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta} ,,, p(\vartheta | d) ,,$$

	Or sample its posterior distribution:

	$$\vartheta^{(k)} \sim p(\vartheta | d).$$

### **Download Korali**

+ Check the following [requirement list](requirements.md) to make sure your system is ready to build Korali:

+ Download and Install Korali
Download Korali's latest distribution into your system. 

	```shell
	wget 'https://www.cse-lab.ethz.ch/korali/distributions/korali-latest.tar.gz'
	```
	
+ Install Korali
Extract Korali and run its installation script.

	```shell
	tar -xzvf korali-latest.tar.gz
	cd korali
	./install
	```