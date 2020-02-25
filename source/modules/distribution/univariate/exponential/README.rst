.. attention::
   This documentation is currently under construction.

*******************************
Exponential Distribution
*******************************

## Exponential
          
The Exponential distribution has the probability density function (PDF):

$$f(x;\beta) = \begin{cases}
\frac{1}{\beta} e^{-\frac{x}{\beta}} & x \ge 0, \\
0 & x < 0.
\end{cases}$$

Where $\beta > 0$ is mean, standard deviation, and scale parameter of the distribution, the reciprocal of the rate parameter, $\lambda$, defined above. In this specification, $\beta$ is a survival parameter in the sense that if a random variable X is the duration of time that a given biological or mechanical system manages to survive and $X ~ Exp(\beta)$ then $E[X] = \beta$.





