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

 js["Minimum"] = _lowerBound;
 js["Maximum"] = _upperBound;

 js["Initial Mean"] = _initialValue;
 js["Initial StdDev"] = _initialStdDev;
 js["Min Std Change"] = _minStdDevChange;

 return js;
}

void Korali::Variable::Base::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING, "Unnamed Variable");

 auto typeString = consume(js, { "Type" }, KORALI_STRING, "Computational");
 if (typeString == "Computational")  _type = KORALI_COMPUTATIONAL;
 if (typeString == "Statistical")    _type = KORALI_STATISTICAL;



 if( isDefined(js, {"Minimum"}) )
  _lowerBound = consume(js, { "Minimum" }, KORALI_NUMBER);
 else
 {
  _lowerBound = -INFINITY;
  fprintf(stderr,"[Korali] Warning: Minimum for \'%s\' not set, initializing to %.4f.\n", _name.c_str(), _lowerBound);
 }


 if( isDefined(js, {"Maximum"}) )
  _upperBound = consume(js, { "Maximum" }, KORALI_NUMBER);
 else
 {
  _upperBound = INFINITY;
  fprintf(stderr,"[Korali] Warning: Maximum for \'%s\' not set, initializing to %.4f.\n", _name.c_str(), _upperBound);
 }


 if( _upperBound <= _lowerBound )
 {
  fprintf(stderr, "[Korali] Error: Invalid Minimum/Maximum for %s.\n", _name.c_str());
  exit(-1);
 }



 if( isDefined(js, {"Initial Mean"}) )
 {
  _initialValue  = consume(js, { "Initial Mean" }, KORALI_NUMBER);
  if(_initialValue < _lowerBound || _initialValue > _upperBound)
  {
   fprintf(stderr,"[Korali] Error: Initial Mean (%.4f) for \'%s\' is out of bounds (%.4f-%.4f).\n", _initialValue, _name.c_str(), _lowerBound, _upperBound);
   exit(-1);
  }
 }
 else
 {
   if( isinf(_upperBound) || isinf(_lowerBound) )
   {
     fprintf(stderr, "[Korali] Error: Minimum and/or Maximum is infinite. Define Initial Mean for %s.\n", _name.c_str());
     exit(-1);
   }
   _initialValue = 0.5*(_lowerBound+_upperBound);
   fprintf(stderr,"[Korali] Warning: Initial Mean for \'%s\' not set, initializing to %.4f.\n", _name.c_str(), _initialValue);
 }



 if( isDefined(js, {"Initial StdDev"}) )
 {
  _initialStdDev = consume(js, { "Initial StdDev" }, KORALI_NUMBER);
 }
 else
 {
   if( isinf(_upperBound) || isinf(_lowerBound) )
   {
     fprintf(stderr, "[Korali] Error: Minimum and/or Maximum is infinite. Define Initial StdDev for %s.\n", _name.c_str());
     exit(-1);
   }
   _initialStdDev = 0.3*(_upperBound-_lowerBound);
   fprintf(stderr,"[Korali] Warning: Initial StdDev for \'%s\' not set, initializing to %.4f.\n", _name.c_str(), _initialStdDev);
 }



 _minStdDevChange = consume(js, { "Min Std Change" }, KORALI_NUMBER, "0.0");


 std::cout << "min = " << _lowerBound << std::endl;
 std::cout << "max = " << _upperBound << std::endl;
 std::cout << "mean = " << _initialValue << std::endl;
 std::cout << "sd   = " << _initialStdDev << std::endl;
 exit(1);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/
