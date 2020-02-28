Preserving Quantities of Interest
=====================================================

In this tutorial we show how important outputs from the computational model can be stored,
that otherwise would be lost.

Example Scripts
---------------------------
    + *run-mcmc.py*

Computational Model
---------------------------

In the model we assign QoI to the Korali `Sample`:
::
    # Store QoI
    d["Apples"]  = a
    d["Bananas"] = b


Execute
---------------------------

In Korali we have to set this flag to store the values of `Apples` and `Bananas`:
::
    e["File Output"]["Store Samples"] = True

All evaluations can be found in the files in `_korali_results`.
