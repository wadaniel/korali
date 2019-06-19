#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Single::Single(nlohmann::json& js)
{
 setConfiguration(js);
 _currentSample = 0;
}

Single::~Single()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Single::getConfiguration(nlohmann::json& js)
{
 js["Conduit"] = "Single";
}

void Single::setConfiguration(nlohmann::json& js)
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Single::run()
{
 _k->_solver->run();
}

void Single::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::ModelData data;

 _k->_problem->packVariables(&sampleArray[_k->N*sampleId], data);

 data._hashId = _currentSample++;

 _k->_model(data);

 double fitness = _k->_problem->evaluateFitness(data);
 _k->_solver->processSample(sampleId, fitness);
}

void Single::checkProgress()
{

}

bool Single::isRoot()
{
 return true;
}
