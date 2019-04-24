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

 js["Objective"] = "Direct Evaluation";

 return js;
}

void Korali::Problem::Direct::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

double Korali::Problem::Direct::evaluateFitness(double* sample)
{
 if (_k->_statisticalParameterCount != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct Evaluation problem requires 0 statistical parameters.\n");
  exit(-1);
 }

 if (isSampleOutsideBounds(sample)) return -DBL_MAX;

 modelData d;
 for (size_t i = 0; i < _k->N; i++) d._parameters.push_back(sample[i]);
 _k->_model(d);

 if (d._results.size() != 1)
 {
  fprintf(stderr, "[Korali] Error: The direct evaluation problem requires exactly a 1-element result array.\n");
  fprintf(stderr, "[Korali]        Provided: %lu.\n", d._results.size());
  exit(-1);
 }

 return d._results[0];
}
