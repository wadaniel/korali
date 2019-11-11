#include "korali.hpp"
#include "model/jacobi.h"

int main(int argc, char* argv[])
{
 MPI_Init(&argc, &argv);

 auto m = korali::Engine();

 m["Problem"]["Type"] = "Evaluation/Bayesian/Inference/Reference";
 m["Problem"]["Likelihood Model"] = "Additive Normal";
 m["Problem"]["Reference Data"] = getReferenceData();
 m["Problem"]["Computational Model"] = &jacobi;

 m["Distributions"][0]["Name"] = "Uniform 0";
 m["Distributions"][0]["Type"] = "Univariate/Uniform";
 m["Distributions"][0]["Minimum"] = +0.0;
 m["Distributions"][0]["Maximum"] = +5.0;

 m["Distributions"][1]["Name"] = "Uniform 1";
 m["Distributions"][1]["Type"] = "Univariate/Uniform";
 m["Distributions"][1]["Minimum"] = -5.0;
 m["Distributions"][1]["Maximum"] = +0.0;

 m["Distributions"][2]["Name"] = "Uniform 2";
 m["Distributions"][2]["Type"] = "Univariate/Uniform";
 m["Distributions"][2]["Minimum"] = +1.0;
 m["Distributions"][2]["Maximum"] = +6.0;

 m["Distributions"][3]["Name"] = "Uniform 3";
 m["Distributions"][3]["Type"] = "Univariate/Uniform";
 m["Distributions"][3]["Minimum"] = -4.0;
 m["Distributions"][3]["Maximum"] = +1.0;

 m["Distributions"][4]["Name"] = "Uniform 4";
 m["Distributions"][4]["Type"] = "Univariate/Uniform";
 m["Distributions"][4]["Minimum"] = +2.0;
 m["Distributions"][4]["Maximum"] = +7.0;

 m["Distributions"][5]["Name"] = "Uniform 5";
 m["Distributions"][5]["Type"] = "Univariate/Uniform";
 m["Distributions"][5]["Minimum"] = -3.0;
 m["Distributions"][5]["Maximum"] = +2.0;

 m["Distributions"][6]["Name"] = "Uniform 6";
 m["Distributions"][6]["Type"] = "Univariate/Uniform";
 m["Distributions"][6]["Minimum"] = 0.0;
 m["Distributions"][6]["Maximum"] = +20.0;

 m["Variables"][0]["Name"] = "X0";
 m["Variables"][0]["Bayesian Type"] = "Computational";
 m["Variables"][0]["Prior Distribution"] = "Uniform 0";

 m["Variables"][1]["Name"] = "X1";
 m["Variables"][1]["Bayesian Type"] = "Computational";
 m["Variables"][1]["Prior Distribution"] = "Uniform 1";

 m["Variables"][2]["Name"] = "Y0";
 m["Variables"][2]["Bayesian Type"] = "Computational";
 m["Variables"][2]["Prior Distribution"] = "Uniform 2";

 m["Variables"][3]["Name"] = "Y1";
 m["Variables"][3]["Bayesian Type"] = "Computational";
 m["Variables"][3]["Prior Distribution"] = "Uniform 3";

 m["Variables"][4]["Name"] = "Z0";
 m["Variables"][4]["Bayesian Type"] = "Computational";
 m["Variables"][4]["Prior Distribution"] = "Uniform 4";

 m["Variables"][5]["Name"] = "Z1";
 m["Variables"][5]["Bayesian Type"] = "Computational";
 m["Variables"][5]["Prior Distribution"] = "Uniform 5";

 m["Variables"][6]["Name"] = "Sigma";
 m["Variables"][6]["Bayesian Type"] = "Statistical";
 m["Variables"][6]["Prior Distribution"] = "Uniform 6";

 m["Solver"]["Type"] = "Sampler/TMCMC";
 m["Solver"]["Covariance Scaling"] = 0.02;
 m["Solver"]["Population Size"] = 100;
 m["Solver"]["Termination Criteria"]["Max Generations"] = 20;

 auto k = korali::Korali();
 if (argc != 2) { printf("Error: this example requires 'Ranks Per Team' passed as argument.\n"); exit(-1); }

 k["Conduit"]["Type"] = "MPI";
 k["Conduit"]["Ranks Per Team"] = atoi(argv[1]);

 k.run(m);

 return 0;
}
