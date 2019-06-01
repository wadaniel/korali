#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Nonintrusive::Nonintrusive(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);
}

Nonintrusive::~Nonintrusive()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Nonintrusive::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = "Nonintrusive";
 js["Launch Command"] = _command;

 return js;
}

void Nonintrusive::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);

 _command =  consume(js, { "Launch Command" }, KORALI_STRING);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Nonintrusive::run()
{
 _k->_solver->run();
}

void Nonintrusive::evaluateSample(double* sampleArray, size_t sampleId)
{
 Korali::ModelData data;

 int curVar = 0;
 for (; curVar < _k->_problem->_computationalVariableCount; curVar++) data._computationalVariables.push_back(sampleArray[_k->_problem->N*sampleId + curVar]);
 for (; curVar < _k->_problem->_statisticalVariableCount;   curVar++) data._statisticalVariables.push_back(  sampleArray[_k->_problem->N*sampleId + curVar]);

 // _k->_model(data);

 double fitness = _k->_problem->evaluateFitness(data);
 //_k->_solver->processSample(sampleId, fitness);
}

void Nonintrusive::checkProgress()
{

}

bool Nonintrusive::isRoot()
{
 return true;
}
