Surrogate Modelling with Gaussian Processes
=====================================================

In this tutorial we show how to create a surrogate with Gaussian Processes.

Example Scripts
---------------------------
    + *run-gp.py*

Init
---------------------------

As preparation for this tutorial we need to run the MATLAB file `/data/make_data.m` to create synthetic data.


Train
---------------------------

Data is read from a file `data/sincos1d_train.dat` and we fit a GP with 
isotropic Gaussian kernel with a constant additive noise term.
::
    x, y = read_matrix_for_gp('data/sincos1d_train.dat')
    e0 = korali.Experiment()
    e0["Problem"]["Type"] = "Evaluation/GaussianProcess"
    e0["Problem"]["Covariance Function"] = "CovSum ( CovSEiso, CovNoise)"
    e0["Problem"]["X Data"] = x
    e0["Problem"]["Y Data"] = y
    e0["Solver"]["Type"] = "Rprop"
    e0["Solver"]["Termination Criteria"]["Max Generations"] = 200
    e0["Solver"]["Termination Criteria"]["Parameter Relative Tolerance"] = 1e-8
    e0["Console Output"]["Verbosity"] = "Normal"
    e0["Console Output"]["Frequency"] = 10
    e0["File Output"]["Frequency"] = 100
    e0["File Output"]["Path"] = "_korali_result_train"

Test
---------------------------

We evaluate the fitted GP on the data set `data/sincos1d_test.dat`. For this
we use the Korali `Executor` solver.
::
    x, y = read_matrix_for_gp('data/sincos1d_test.dat')
    e1 = korali.Experiment()
    e1["Problem"]["Type"] = "Execution/GaussianProcess"
    e1["Problem"]["Gaussian Process Json File"] =  "_korali_result_train/final.json"
    e1["Problem"]["X Data"] = x
    e1["Problem"]["Y Data"] = y
    e1["Solver"]["Type"] = "Executor"
    e1["Solver"]["Executions Per Generation"] = 1
    e1["Console Output"]["Verbosity"] = "Normal"
    e1["Console Output"]["Frequency"] = 10
    e1["File Output"]["Frequency"] = 100
    e1["File Output"]["Path"] = "_korali_result_test"

Predict
---------------------------

And finally we predict values for data  `data/sincos1d_new.dat`.
::
    x, y = read_matrix_for_gp('data/sincos1d_new.dat',lastColumnIsData=True)
    e2 = korali.Experiment()
    e2["Problem"]["Type"] = "Execution/GaussianProcess"
    e2["Problem"]["Gaussian Process Json File"] =  "_korali_result_train/final.json"
    e2["Problem"]["X Data"] = x
    e2["Problem"]["Y Data"] = y
    e2["Solver"]["Type"] = "Executor"
    e2["Solver"]["Executions Per Generation"] = 1
    e2["Console Output"]["Verbosity"] = "Normal"
    e2["Console Output"]["Frequency"] = 10
    e2["File Output"]["Frequency"] = 100
    e2["File Output"]["Path"] = "_korali_result_new"

Run
---------------------------
All three steps can be run with `.run-gp.py`
