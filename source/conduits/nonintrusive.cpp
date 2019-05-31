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
 double fitness = _k->_problem->evaluateFitness(&sampleArray[_k->_problem->N*sampleId], true);
 _k->_solver->processSample(sampleId, fitness);
}

void Nonintrusive::checkProgress()
{

}

bool Nonintrusive::isRoot()
{
 return true;
}
