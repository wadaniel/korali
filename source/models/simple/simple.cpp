#include "korali.h"

using namespace Korali::Model;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Simple::Simple(std::function<void(Korali::Model::Simple&)> method) : Base::Base()
{
 _method = method;
}

Simple::~Simple()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Simple::getConfiguration()
{
 auto js = this->Base::getConfiguration();

 js["Type"] = "Simple";

 return js;
}

void Simple::setConfiguration(nlohmann::json& js)
{
 this->Base::setConfiguration(js);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void Simple::evaluate(std::vector<double>& parameters, std::vector<double>& results)
{
 Simple evaluator = *this;
 evaluator._self = &evaluator;
 evaluator._parameters = parameters;

 _method(evaluator);

 results = evaluator._results;
}

void Simple::initialize(size_t nRanks)
{
  Base::initialize(nRanks);
}

size_t Simple::getTeamCount()
{
  return _nRanks;
}

size_t Simple::getRanksPerTeam()
{
 return 1;
}
