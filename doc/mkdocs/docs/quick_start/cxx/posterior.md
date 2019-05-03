## Problem setup
In this example we will solve the inverse problem of estimating the parameters
of a linear model using noisy data. We consider the computational model,

$$
f(x;\vartheta) = \vartheta_0 + \vartheta_1 x \,,
$$

for $x\in\mathbb{R}$. We assume the following error model,

$$
y = f(x;\vartheta) + \varepsilon \,,
$$

with $\varepsilon$ a random variable that follows normal distribution with zero
mean and $\sigma$ standard deviation. This assumption leads to the likelihood
function,

$$
p(y|\varphi;x) = \mathcal{N} ( \,y \,| \, f(x;\vartheta), \sigma^2 \,) \,.
$$

where $\varphi=(\vartheta,\sigma)$ is the parameter vector that contains the
computational parameters and the parameters of the statistical model.

We assume that we are given the following data set:

<center>

| $x_i$  | $y_i$  |
|:-:|:-:|
| 1  | 3.2069  |
| 2  | 4.1454  |
| 3  | 4.9393  |
| 4  | 6.0588  |
| 5  | 6.8425  |

</center>

We call this data set $d=\{x_i,y_i\}_{i=1}^5$. Assuming that each datum is
independent, the likelihood of $d$ under the linear model is given by

$$
p(y|\vartheta;x) = \prod_{i=1}^6 \mathcal{N} ( \,y_i \,| \, f(x_i;\vartheta), \sigma^2 \,) \,.
$$

In order to identify the distribution of $\varphi$ conditioned on the observations $d$
we use Bayes' theorem

$$
p(\varphi | y;x) = \frac{ p(y|\varphi;x) \, p(\varphi) }{ p(y) } \, .
$$


As a prior information we choose the uniform distribution in $[-10,10]$ for $\vartheta$
and the uniform distribution in $[0,10]$ for $\sigma$.




## Optimize

### The computational model

```cpp
 void F( std::vector<double>& par, std::vector<double>& result, std::vector<double> x ){
   for( size_t i=0; i<x.size(); i++){
     double res = par[0]*x[i] + par[1];
     result.push_back( res );
   }
 }
```

### The data
```cpp
  std::vector<double> x, y;
  x.push_back(1.0);   y.push_back(3.2069);
  x.push_back(2.0);   y.push_back(4.1454);
  x.push_back(3.0);   y.push_back(4.9393);
  x.push_back(4.0);   y.push_back(6.0588);
  x.push_back(5.0);   y.push_back(6.8425);
```

### The Korali object
```cpp
  auto Fx = [x]( Korali::modelData& d ) {
                F(d.getParameters(), d.getResults(), x);
  };

  auto korali = Korali::Engine( Fx );
```

### The problem type
```cpp
  korali["Problem"]["Objective"] = "Posterior";
```

```cpp
  for (size_t i = 0; i < d.size(); i++)
    korali["Problem"]["Reference Data"][i] = y[i];
```

### The parameters
```cpp
  korali["Parameters"][0]["Name"] = "a";
  korali["Parameters"][0]["Type"] = "Computational";
  korali["Parameters"][0]["Distribution"] = "Uniform";
  korali["Parameters"][0]["Minimum"] = -10.0;
  korali["Parameters"][0]["Maximum"] = +10.0;

  korali["Parameters"][1]["Name"] = "b";
  korali["Parameters"][1]["Type"] = "Computational";
  korali["Parameters"][1]["Distribution"] = "Uniform";
  korali["Parameters"][1]["Minimum"] = -10.0;
  korali["Parameters"][1]["Maximum"] = +10.0;
```

```cpp
  korali["Parameters"][2]["Name"] = "Sigma";
  korali["Parameters"][2]["Type"] = "Statistical";
  korali["Parameters"][2]["Distribution"] = "Uniform";
  korali["Parameters"][2]["Minimum"] = 0.0;
  korali["Parameters"][2]["Maximum"] = 20.0;
```

### The solver
```cpp
  korali["Solver"]["Method"] = "CMA-ES";
  korali["Solver"]["Lambda"] = 12;
  korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
  korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
  korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
  korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;
```

### Run
```cpp
  korali["Seed"] = 0xC0FFEE;
  korali["Verbosity"] = "Normal";

  korali.run();
```






##Sample
```cpp
  korali["Solver"]["Method"] = "TMCMC";
  korali["Solver"]["Covariance Scaling"] = 0.02;
  korali["Solver"]["Population Size"] = 5000;
  korali["Solver"]["Burn In"] = 5;
  korali["Solver"]["Coefficient of Variation"] = 0.5;
```
