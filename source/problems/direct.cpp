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

 js["Type"] = "Direct";

 return js;
}

void Korali::Problem::Direct::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

double Korali::Problem::Direct::evaluateFitness(double* sample, bool isLeader
#ifdef _KORALI_USE_MPI
, MPI_Comm comm
#endif
)
{
 if (_statisticalParameterCount != 0)
 {
  fprintf(stderr, "[Korali] Error: Direct problem requires 0 statistical parameters.\n");
  exit(-1);
 }

 if (isSampleOutsideBounds(sample)) return -DBL_MAX;

 Korali::ModelData data;

 #ifdef _KORALI_USE_MPI
 data._comm = comm;
 #endif

 for (size_t i = 0; i < N; i++) data._parameters.push_back(sample[i]);

 _k->_model(data);

 if (isLeader)
 {
  if (data._results.size() != 1)
  {
   fprintf(stderr, "[Korali] Error: The direct problem requires exactly a 1-element result array.\n");
   fprintf(stderr, "[Korali]        Provided: %lu.\n", data._results.size());
   exit(-1);
  }

  return data._results[0];
 }

 return 0;
}

double Korali::Problem::Direct::evaluateLogPrior(double* sample)
{
  return 0.0;
}
