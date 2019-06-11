#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Direct::Direct(nlohmann::json& js) : Korali::Problem::Base::Base(js)
{
 setConfiguration(js);
}

Korali::Problem::Direct::~Direct()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::Direct::getConfiguration()
{
 auto js = this->Korali::Problem::Base::getConfiguration();

 js["Evaluation Type"] = "Direct";

 return js;
}

void Korali::Problem::Direct::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

double Korali::Problem::Direct::evaluateFitness(Korali::ModelData& data)
{
 if (_statisticalVariableCount != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct problem requires 0 statistical parameters.\n");
  exit(-1);
 }

 if (data._results.size() != 1)
 {
  fprintf(stderr, "[Korali] Error: The direct problem requires exactly a 1-element result array.\n");
  fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
  exit(-1);
 }

 return data._results[0];
}

double Korali::Problem::Direct::evaluateLogPrior(double* sample)
{
  return 0.0;
}
