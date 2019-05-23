#include "korali.h"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);
 auto korali = Korali::Engine();

 korali["Seed"] = 0xC0FFEE;
 korali["Verbosity"] = "Detailed";

 korali["Problem"]["Type"] = "Bayesian";
 korali["Problem"]["Variables"][0]["Name"] = "X0";
 korali["Problem"]["Variables"][0]["Type"] = "Computational";
 korali["Problem"]["Variables"][0]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][0]["Minimum"] = +0.0;
 korali["Problem"]["Variables"][0]["Maximum"] = +5.0;

 korali["Problem"]["Variables"][1]["Name"] = "X1";
 korali["Problem"]["Variables"][1]["Type"] = "Computational";
 korali["Problem"]["Variables"][1]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][1]["Minimum"] = -5.0;
 korali["Problem"]["Variables"][1]["Maximum"] = +0.0;

 korali["Problem"]["Variables"][2]["Name"] = "Y0";
 korali["Problem"]["Variables"][2]["Type"] = "Computational";
 korali["Problem"]["Variables"][2]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][2]["Minimum"] = +1.0;
 korali["Problem"]["Variables"][2]["Maximum"] = +6.0;

 korali["Problem"]["Variables"][3]["Name"] = "Y1";
 korali["Problem"]["Variables"][3]["Type"] = "Computational";
 korali["Problem"]["Variables"][3]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][3]["Minimum"] = -4.0;
 korali["Problem"]["Variables"][3]["Maximum"] = +1.0;

 korali["Problem"]["Variables"][4]["Name"] = "Z0";
 korali["Problem"]["Variables"][4]["Type"] = "Computational";
 korali["Problem"]["Variables"][4]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][4]["Minimum"] = +2.0;
 korali["Problem"]["Variables"][4]["Maximum"] = +7.0;

 korali["Problem"]["Variables"][5]["Name"] = "Z1";
 korali["Problem"]["Variables"][5]["Type"] = "Computational";
 korali["Problem"]["Variables"][5]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][5]["Minimum"] = -3.0;
 korali["Problem"]["Variables"][5]["Maximum"] = +2.0;

 korali["Problem"]["Variables"][6]["Name"] = "Sigma";
 korali["Problem"]["Variables"][6]["Type"] = "Statistical";
 korali["Problem"]["Variables"][6]["Distribution"] = "Uniform";
 korali["Problem"]["Variables"][6]["Minimum"] = 0.0;
 korali["Problem"]["Variables"][6]["Maximum"] = 20.0;

 korali["Problem"]["Reference Data"] += 0.978724;
 korali["Problem"]["Reference Data"] += 0.960720;
 korali["Problem"]["Reference Data"] += 0.738612;
 korali["Problem"]["Reference Data"] += 0.713227;
 korali["Problem"]["Reference Data"] += 0.988677;
 korali["Problem"]["Reference Data"] += 1.001350;
 korali["Problem"]["Reference Data"] += 1.001672;
 korali["Problem"]["Reference Data"] += 0.974431;

 korali["Solver"]["Method"] = "CMA-ES";
 korali["Solver"]["Lambda"] = 16;
 korali["Solver"]["Termination Criteria"]["Min DeltaX"] = 1e-4;
 korali["Solver"]["Termination Criteria"]["Fitness Diff Threshold"] = 1e-4;
 korali["Solver"]["Termination Criteria"]["Max Condition Covariance"] = 1e10;
 
 korali["Conduit"]["Ranks Per Team"] = 2;

 std::vector<double> xdata;

 xdata.push_back(0.364); xdata.push_back(0.944); xdata.push_back(0.560);
 xdata.push_back(0.151); xdata.push_back(0.863); xdata.push_back(0.983);
 xdata.push_back(0.292); xdata.push_back(0.979); xdata.push_back(0.876);
 xdata.push_back(0.451); xdata.push_back(0.014); xdata.push_back(0.150);
 xdata.push_back(0.893); xdata.push_back(0.935); xdata.push_back(0.316);
 xdata.push_back(0.093); xdata.push_back(0.287); xdata.push_back(0.921);
 xdata.push_back(0.671); xdata.push_back(0.565); xdata.push_back(0.078);
 xdata.push_back(0.565); xdata.push_back(0.779); xdata.push_back(0.974);

 korali.run([xdata](Korali::ModelData& d) { jacobi(xdata, d.getParameters(), d.getResults(), d.getComm()); });

 // Expected Solution:
 // X0 = +3.71
 // X1 = -2.34
 // Y0 = +1.32
 // Y1 = +0.78
 // Z0 = +2.25
 // Z1 = +1.81

 return 0;
}
