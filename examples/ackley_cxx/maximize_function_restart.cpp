#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();
 korali.loadState("_korali_result/s00005.json");

 korali.setModel([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 korali.run();

 return 0;
}
