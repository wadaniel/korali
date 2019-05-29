#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Solver::Base::Base(nlohmann::json& js)
{
 setConfiguration(js);
}

Korali::Solver::Base::~Base()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Solver::Base::getConfiguration()
{
 auto js = nlohmann::json();

 return js;
}

void Korali::Solver::Base::setConfiguration(nlohmann::json& js)
{

}

void Korali::Solver::Base::setState(nlohmann::json& js)
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/




