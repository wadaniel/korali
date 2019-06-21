

In this tutorial we show how to restart an **optimization** algorithm.


The code for this tutorial in **Python** can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/b1-restart).


First delete the `_korali_result` folder and run the optimization example from [this](../a1-optimization/optimization.md) tutorial. The code can be found [here](https://github.com/cselab/skorali/blob/master/tutorials/a1-optimization).


Assume that we want to resume the algorithm from the 5th generation. We just have to run the following script,
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
