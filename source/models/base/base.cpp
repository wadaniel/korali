#include "korali.h"

using namespace Korali::Model;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Base::Base(nlohmann::json& js)
{
 setConfiguration(js);
}

Base::~Base()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Base::getConfiguration()
{
 auto js = nlohmann::json();
 return js;
}

void Base::setConfiguration(nlohmann::json& js)
{
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

