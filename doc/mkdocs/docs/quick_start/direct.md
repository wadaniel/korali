In this section we will present how to optimize or sample from a given function $f(\vartheta)=\vartheta^2$ for $\vartheta\in[-10,10]$.

## Optimize

First include the Korali header
```cpp
#include "korali.h"
```

Then write the code for $f$
```cpp
void F(std::vector<double>& x, std::vector<double>& result){
  result.push_back( -pow( x[0]-2. , 2) );
}
```





Next we construct a `korali` object by
```cpp
auto korali = Korali::Engine([](Korali::modelData& d) {
  F01(d.getParameters(), d.getResults());
});
```

The we set the type of the problem to `Direct Evaluation`
```cpp
korali["Problem"]["Objective"] = "Direct Evaluation";
```

```cpp
korali["Parameters"][0]["Name"] = "X0";
korali["Parameters"][0]["Type"] = "Computational";
korali["Parameters"][0]["Distribution"] = "Uniform";
korali["Parameters"][0]["Minimum"] = -10.0;
korali["Parameters"][0]["Maximum"] = +10.0;
```


```cpp
korali["Solver"]["Method"] = "CMA-ES";
korali["Solver"]["Lambda"] = 12;
korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-11;
korali["Solver"]["Termination Criteria"]["Min Fitness"] = 1e-12;
korali["Solver"]["Termination Criteria"]["Max Generations"] = 1e4;
korali["Solver"]["Termination Criteria"]["Max Model Evaluations"] = 1e4;
```


```cpp
korali["Seed"] = 0xC0FFEE;
korali["Verbosity"] = "Detailed";
```


```cpp
korali.run();
```






## Sample
