#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Problem::Base::Base(nlohmann::json& js)
{
 setConfiguration(js);
}

Korali::Problem::Base::~Base()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Problem::Base::getConfiguration()
{
 auto js = nlohmann::json();

 return js;
}

void Korali::Problem::Base::setConfiguration(nlohmann::json& js)
{

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

bool Korali::Problem::Base::isSampleOutsideBounds(double* sample)
{
  for (size_t i = 0; i < _k->N; i++)
  	if (sample[i] < _k->_parameters[i]->_lowerBound || sample[i] > _k->_parameters[i]->_upperBound) return true;
  return false;
}
