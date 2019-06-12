#include "korali.h"
#include <chrono>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Variable::Base::Base(nlohmann::json& js, size_t seed) : Korali::Variable::Base::Base(seed)
{
 setConfiguration(js);
}

Korali::Variable::Base::Base(size_t seed)
{
 _range = gsl_rng_alloc (gsl_rng_default);
 gsl_rng_set(_range, seed);
}

Korali::Variable::Base::~Base()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Variable::Base::getConfiguration()
{
 auto js = nlohmann::json();

 js["Name"] = _name;

 if(_type == KORALI_COMPUTATIONAL) js["Type"] = "Computational";
 if(_type == KORALI_STATISTICAL)   js["Type"] = "Statistical";


 return js;
}

void Korali::Variable::Base::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING);

 auto typeString = consume(js, { "Type" }, KORALI_STRING, "Computational");
 if (typeString == "Computational")  _type = KORALI_COMPUTATIONAL;
 if (typeString == "Statistical")    _type = KORALI_STATISTICAL;

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/
