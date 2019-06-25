In this tutorial we show how to restart a previous Korali run.

!!! info
	An example Python code for this tutorial can be found in:
	[tutorials/b1-restart](https://github.com/cselab/skorali/blob/master/tutorials/b1-restart)


Assume that we want to resume a previous Korali run, restarting from the one of the last generations. All we need to do is to run the following script:

```python
#!/usr/bin/env python3
import sys
sys.path.append('./model')
from directModel import *
import korali

k = korali.initialize()
k.loadState("_korali_result/s00498.json")
k.setModel(evaluateModel)
k.run();
```

Korali will automatically load all the configuration and solver state from the provided file, and run from that point until completion.
