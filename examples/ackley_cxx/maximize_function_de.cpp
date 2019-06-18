#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto k = Korali::Engine();

 k["Problem"] = "Direct";
 k["Solver"]  = "DE";
 
 k["Variables"][0]["Name"] = "X0";
 k["Variables"][0]["DE"]["Lower Bound"] = -32.0;
 k["Variables"][0]["DE"]["Upper Bound"] = +32.0; 
 
 k["Variables"][1]["Name"] = "X1";
 k["Variables"][1]["DE"]["Lower Bound"] = -32.0;
 k["Variables"][1]["DE"]["Upper Bound"] = +32.0;
 
 k["Variables"][2]["Name"] = "X2";
 k["Variables"][2]["DE"]["Lower Bound"] = -32.0;
 k["Variables"][2]["DE"]["Upper Bound"] = +32.0;

 k["Variables"][3]["Name"] = "X3";
 k["Variables"][3]["DE"]["Lower Bound"] = -32.0;
 k["Variables"][3]["DE"]["Upper Bound"] = +32.0;

 k["DE"]["Objective"] = "Maximize";
 k["DE"]["Sample Count"] = 40;
 k["DE"]["Termination Criteria"]["Max Generations"]["Value"] = 100;
 
 
 k.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 k.run();

 return 0;
}
