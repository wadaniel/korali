#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Single::Single(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);
}

Single::~Single()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Single::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = "Single";

 return js;
}

void Single::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);
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
 double fitness = _k->_problem->evaluateFitness(&sampleArray[_k->_problem->N*sampleId], true, MPI_COMM_NULL);
 _k->_solver->processSample(sampleId, fitness);
}

void Single::checkProgress()
{

}

bool Single::isRoot()
{
 return true;
}
