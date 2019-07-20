#include "korali.h"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);
 auto k = Korali::Engine();

 k["Problem"]["Type"] = "Bayesian Inference";
 k["Problem"]["Likelihood"]["Model"] = "Additive Gaussian";
 k["Problem"]["Likelihood"]["Reference Data"] = getReferenceData();

 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["Type"] = "Computational";
 k["Variables"][0]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][0]["Prior Distribution"]["Minimum"] = +0.0;
 k["Variables"][0]["Prior Distribution"]["Maximum"] = +5.0;

 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["Type"] = "Computational";
 k["Variables"][1]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][1]["Prior Distribution"]["Minimum"] = -5.0;
 k["Variables"][1]["Prior Distribution"]["Maximum"] = +0.0;

 k["Variables"][2]["Name"] = "Y0";
 k["Variables"][2]["Type"] = "Computational";
 k["Variables"][2]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][2]["Prior Distribution"]["Minimum"] = +1.0;
 k["Variables"][2]["Prior Distribution"]["Maximum"] = +6.0;

 k["Variables"][3]["Name"] = "Y1";
 k["Variables"][3]["Type"] = "Computational";
 k["Variables"][3]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][3]["Prior Distribution"]["Minimum"] = -4.0;
 k["Variables"][3]["Prior Distribution"]["Maximum"] = +1.0;

 k["Variables"][4]["Name"] = "Z0";
 k["Variables"][4]["Type"] = "Computational";
 k["Variables"][4]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][4]["Prior Distribution"]["Minimum"] = +2.0;
 k["Variables"][4]["Prior Distribution"]["Maximum"] = +7.0;

 k["Variables"][5]["Name"] = "Z1";
 k["Variables"][5]["Type"] = "Computational";
 k["Variables"][5]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][5]["Prior Distribution"]["Minimum"] = -3.0;
 k["Variables"][5]["Prior Distribution"]["Maximum"] = +2.0;

 k["Variables"][6]["Name"] = "Sigma";
 k["Variables"][6]["Type"] = "Statistical";
 k["Variables"][6]["Prior Distribution"]["Type"] = "Uniform";
 k["Variables"][6]["Prior Distribution"]["Minimum"] = 0.0;
 k["Variables"][6]["Prior Distribution"]["Maximum"] = 20.0;

 k["Solver"]["Type"] = "TMCMC";
 k["Solver"]["Covariance Scaling"] = 0.02;
 k["Solver"]["Population Size"] = 100;

 k["Conduit"]["MPI"]["Enabled"] = true;
 k["Conduit"]["MPI"]["Ranks Per Team"] = atoi(argv[1]);

 k.setModel([](Korali::Model& d) { jacobi(getPointData(), d.getVariables(), d.getResults(), d.getComm()); });

 k.run();

 return 0;
}
