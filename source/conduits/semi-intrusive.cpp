#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void SemiIntrusive::getConfiguration()
{
 _k->_js["Conduit"] = "Semi-Intrusive";
}

void SemiIntrusive::setConfiguration()
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void SemiIntrusive::initialize()
{
	_currentSample = 0;
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
