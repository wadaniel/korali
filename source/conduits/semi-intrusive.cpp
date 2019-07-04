#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

SemiIntrusive::SemiIntrusive(nlohmann::json& js)
{
 setConfiguration(js);
 _currentSample = 0;
}

SemiIntrusive::~SemiIntrusive()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void SemiIntrusive::getConfiguration(nlohmann::json& js)
{
 js["Conduit"] = "Semi-Intrusive";
}

void SemiIntrusive::setConfiguration(nlohmann::json& js)
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void SemiIntrusive::initialize()
{
}

void SemiIntrusive::finalize()
{
}

void SemiIntrusive::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::ModelData data;

 _k->_problem->packVariables(&sampleArray[_k->N*sampleId], data);

 data._hashId = _currentSample++;

 _k->_model(data);

 double fitness = _k->_problem->evaluateFitness(data);
 _k->_solver->processSample(sampleId, fitness);
}

void SemiIntrusive::checkProgress()
{

}

bool SemiIntrusive::isRoot()
{
 return true;
}
