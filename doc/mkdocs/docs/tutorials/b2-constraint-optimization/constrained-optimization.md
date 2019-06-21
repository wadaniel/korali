

In this tutorial we show how to **optimize** and **sample** the posterior
distribution of a Bayesian inference problem.


The code for this tutorial in **Python** can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/b1-restart).


```python
#!/usr/bin/env python3
import sys
sys.path.append('./model')
from directModel import *
import korali


k2 = korali.Engine()

k2.loadState("_korali_result/s00005.json")

k2.setModel(evaluateModel)

k2.run();
```
