#include "korali.h"

using namespace Korali::Model;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Sequential::Sequential(std::function<void(Korali::Model::Sequential&)> method) : Base::Base()
{
 _method = method;
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


void Sequential::evaluate(std::vector<double>& parameters, std::vector<double>& results)
{
 Sequential evaluator = *this;
 evaluator._self = &evaluator;
 evaluator._parameters = parameters;

 _method(evaluator);

 results = evaluator._results;
}
