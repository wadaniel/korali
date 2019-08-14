#include "korali.hpp"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);
 auto k = Korali::Engine();

 if (argc != 2) { printf("Error: this example requires 'Ranks Per Team' passed as argument.\n"); exit(-1); }

 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Normal";
 k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData();

 k["Problem"]["Variables"][0]["Name"] = "X0";
 k["Problem"]["Variables"][0]["Type"] = "Computational";
 k["Problem"]["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][0]["Prior Distribution"]["Minimum"] = +0.0;
 k["Problem"]["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0;

 k["Problem"]["Variables"][1]["Name"] = "X1";
 k["Problem"]["Variables"][1]["Type"] = "Computational";
 k["Problem"]["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0;
 k["Problem"]["Variables"][1]["Prior Distribution"]["Maximum"] = +0.0;

 k["Problem"]["Variables"][2]["Name"] = "Y0";
 k["Problem"]["Variables"][2]["Type"] = "Computational";
 k["Problem"]["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][2]["Prior Distribution"]["Minimum"] = +1.0;
 k["Problem"]["Variables"][2]["Prior Distribution"]["Maximum"] = +6.0;

 k["Problem"]["Variables"][3]["Name"] = "Y1";
 k["Problem"]["Variables"][3]["Type"] = "Computational";
 k["Problem"]["Variables"][3]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][3]["Prior Distribution"]["Minimum"] = -4.0;
 k["Problem"]["Variables"][3]["Prior Distribution"]["Maximum"] = +1.0;

 k["Problem"]["Variables"][4]["Name"] = "Z0";
 k["Problem"]["Variables"][4]["Type"] = "Computational";
 k["Problem"]["Variables"][4]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][4]["Prior Distribution"]["Minimum"] = +2.0;
 k["Problem"]["Variables"][4]["Prior Distribution"]["Maximum"] = +7.0;

 k["Problem"]["Variables"][5]["Name"] = "Z1";
 k["Problem"]["Variables"][5]["Type"] = "Computational";
 k["Problem"]["Variables"][5]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][5]["Prior Distribution"]["Minimum"] = -3.0;
 k["Problem"]["Variables"][5]["Prior Distribution"]["Maximum"] = +2.0;

 k["Problem"]["Variables"][6]["Name"] = "Sigma";
 k["Problem"]["Variables"][6]["Type"] = "Statistical";
 k["Problem"]["Variables"][6]["Prior Distribution"]["Type"] = "Uniform";
 k["Problem"]["Variables"][6]["Prior Distribution"]["Minimum"] = 0.0;
 k["Problem"]["Variables"][6]["Prior Distribution"]["Maximum"] = 20.0;

 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Covariance Scaling"] = 0.02;
 k["Solver"]["Population Size"] = 100;
 k["Solver"]["Termination Criteria"]["Max Generations"] = 20;

 k["Conduit"]["Type"] = "MPI";
 k["Conduit"]["Ranks Per Team"] = atoi(argv[1]);

 k.setReferenceModel([](Korali::Model::Reference& d)
   { jacobi(getPointData(), d.getVariables(), d.getEvaluations(), getKoraliMPIComm()); });

 k.run();

 return 0;
}
