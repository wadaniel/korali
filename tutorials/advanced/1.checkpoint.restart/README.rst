# B.1 - Checkpoint / Resuming a Previous Korali Run

In this tutorial we show how to restart a previous Korali run.

## Steps to Restart Execution

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
