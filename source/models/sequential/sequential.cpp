#include "korali.h"

using namespace Korali::Model;

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

