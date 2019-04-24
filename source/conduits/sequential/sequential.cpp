#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Sequential::Sequential(nlohmann::json& js) : Base::Base(js)
{
 setConfiguration(js);
}

Sequential::~Sequential()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Sequential::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = "Sequential";

 return js;
}

void Sequential::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Sequential::run()
{
 _k->_solver->run();
}

void Sequential::evaluateSample(double* sampleArray, size_t sampleId)
{
 double fitness = _k->_problem->evaluateFitness(&sampleArray[_k->N*sampleId]);
 _k->_solver->processSample(sampleId, fitness);
}

void Sequential::checkProgress()
{

}

bool Sequential::isRoot()
{
 return true;
}
