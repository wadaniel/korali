#include "korali.h"

using namespace Korali::Conduit;

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Base::Base(nlohmann::json& js)
{
 setConfiguration(js);
 _currentSample = 0;
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

