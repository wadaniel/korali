#include "korali.h"
#include "model/ackley.h"

int main(int argc, char* argv[])
{
 auto korali = Korali::Engine();
 korali.loadState("_korali_result/s00005.json");

 korali.run([](Korali::ModelData& d) { m_ackley(d.getVariables(), d.getResults()); });

 return 0;
}
