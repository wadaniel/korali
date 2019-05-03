## Problem setup
In this section we will present how to optimize or sample from a given function $f(\vartheta)=\vartheta^2$ for $\vartheta\in[-10,10]$.

The code for this tutorial can be found [here](https://github.com/cselab/skorali/blob/master/examples/cxx/quick_start/direct.cpp).

## Optimize

First include the Korali header
```cpp
#include "korali.h"
```

### 1. The computational model
Then write the code for $f$
```cpp
void F(std::vector<double>& x, std::vector<double>& result){
  result.push_back( -pow( x[0]-2. , 2) );
}
```


Now we are in `main` and write a lambda function for `F`
```
auto Fx = []( Korali::modelData& d ){ F(d.getParameters(), d.getResults()); };
```

### 2. The korali object

Next we construct a `korali` object using the lambda function
```cpp
auto korali = Korali::Engine( Fx );
```

### 3. The Problem type
The we set the type of the problem to `Direct Evaluation`
```cpp
korali["Problem"]["Objective"] = "Direct Evaluation";
```

### 4. The Parameters
```cpp
korali["Parameters"][0]["Name"] = "X0";
korali["Parameters"][0]["Type"] = "Computational";
korali["Parameters"][0]["Distribution"] = "Uniform";
korali["Parameters"][0]["Minimum"] = -10.0;
korali["Parameters"][0]["Maximum"] = +10.0;
```

### 5. The Solver
```cpp
korali["Solver"]["Method"] = "CMA-ES";
korali["Solver"]["Lambda"] = 12;
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;
```

### 6. Run
```cpp
korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
```


```cpp
korali.run();
```






## Sample

### 1. The Solver

```cpp
korali["Solver"]["Method"] = "TMCMC";
korali["Solver"]["Covariance Scaling"] = 0.02;
korali["Solver"]["Population Size"] = 5000;
korali["Solver"]["Burn In"] = 5;
korali["Solver"]["Coefficient of Variation"] = 0.5;
```
