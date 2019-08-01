# Korali  ![](images/logo.png){: style="height:25px;width:25px"}
<center>**High-performance framework for uncertainty quantification of large-scale models.**</center>

### **What Korali Does**

??? abstract "Optimization"
	Given any objective function $f:\mathbb{R}^N\rightarrow\mathbb{R}$ Korali can find its minima or maxima:

	Given constraints...

??? abstract "Sampling"
	Given any probability distribution of type $f:\mathbb{R}^N\rightarrow\mathbb{R}$, Korali can draw samples to determine its distribution: 
	
	$$\vartheta^{(k)} \sim f.$$

??? abstract "Bayesian Inference"
	Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:

	$$ p(\vartheta | d)  =  \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

	Allowing users to compute the maximum a posteriori estimate:

	$$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  p(\vartheta | d) $$

	Or sample its posterior distribution:

	$$\vartheta^{(k)} \sim p(\vartheta | d).$$
	
??? abstract "Hierarchical Bayesian Modeling"
	Given a set of data $d$, the output of a computational model $f(x;\vartheta)$, a likelihood function $p(d|\vartheta)$, and a prior probability density $p(\vartheta)$ Korali will build the posterior distribution:

	$$ p(\vartheta | d)  =  \frac{p(d | \vartheta) p(\vartheta)}{p(d)} $$

	Allowing users to compute the maximum a posteriori estimate:

	$$\vartheta^\star = \mathop{\arg\min}\limits_{\vartheta}  p(\vartheta | d) $$
	
	Or sample its posterior distribution:

	$$\vartheta^{(k)} \sim p(\vartheta | d).$$

### **Download Korali**

[Latest Release: 28/06/2019 - Korali v1.0.0](install.md)


