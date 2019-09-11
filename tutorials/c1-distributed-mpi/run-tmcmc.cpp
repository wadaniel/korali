#include "korali.hpp"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);
 auto k = Korali::Engine();
 auto referenceModel = [](Korali::Sample& d) { jacobi(getPointData(), d.getSampleData(), d.getResult(), getKoraliMPIComm()); };
 if (argc != 2) { printf("Error: this example requires 'Ranks Per Team' passed as argument.\n"); exit(-1); }

 k["Problem"]["Type"] = "Bayesian Inference (Reference Likelihood)";
 k["Problem"]["Likelihood Model"] = "Additive Normal";
 k["Problem"]["Reference Data"] = getReferenceData();
 k["Problem"]["Computational Model"] = referenceModel;

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["Bayesian Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = +0.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["Bayesian Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = +0.0;

 k["Variables"][2]["Name"] = "Y0";
 k["Variables"][2]["Bayesian Type"] = "Computational";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = +1.0;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = +6.0;

 k["Variables"][3]["Name"] = "Y1";
 k["Variables"][3]["Bayesian Type"] = "Computational";
 k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Prior Distribution"]["Minimum"] = -4.0;
 k["Variables"][3]["Prior Distribution"]["Maximum"] = +1.0;

 k["Variables"][4]["Name"] = "Z0";
 k["Variables"][4]["Bayesian Type"] = "Computational";
 k["Variables"][4]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][4]["Prior Distribution"]["Minimum"] = +2.0;
 k["Variables"][4]["Prior Distribution"]["Maximum"] = +7.0;

 k["Variables"][5]["Name"] = "Z1";
 k["Variables"][5]["Bayesian Type"] = "Computational";
 k["Variables"][5]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][5]["Prior Distribution"]["Minimum"] = -3.0;
 k["Variables"][5]["Prior Distribution"]["Maximum"] = +2.0;

 k["Variables"][6]["Name"] = "Sigma";
 k["Variables"][6]["Bayesian Type"] = "Statistical";
 k["Variables"][6]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][6]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][6]["Prior Distribution"]["Maximum"] = 20.0;

 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Covariance Scaling"] = 0.02;
 k["Solver"]["Population Size"] = 100;
 k["Solver"]["Termination Criteria"]["Max Generations"] = 20;

 k["Conduit"]["Type"] = "MPI";
 k["Conduit"]["Ranks Per Team"] = atoi(argv[1]);

 k.run();

 return 0;
}
