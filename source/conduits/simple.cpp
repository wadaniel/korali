#include "korali.hpp"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

void Simple::initialize()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Simple::getConfiguration()
{
 _k->_js["Conduit"]["Type"] = "Simple";
}

void Simple::setConfiguration()
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Simple::finalize()
{

}

void Simple::evaluateSample(double* sampleArray, size_t sampleId)
{
 _k->functionEvaluationCount++;

  Korali::Model data;

 _k->_problem->packVariables(sampleArray, data);
  data._sampleId = sampleId;
 _k->_model(data);

 double fitness = _k->_problem->evaluateFitness(data);
 _k->functionEvaluationCount++;
 _k->_solver->processSample(sampleId, fitness);
}

void Simple::checkProgress()
{

}

bool Simple::isRoot()
{
 return true;
}

void Simple::abort()
{
 exit(-1);
}
