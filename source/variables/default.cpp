#include "korali.h"
#include <chrono>

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Variable::Default::Default(nlohmann::json& js) : Korali::Variable::Default::Default()
{
 setConfiguration(js);
}

Korali::Variable::Default::Default()
{

}

Korali::Variable::Default::~Default()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

void Korali::Variable::Default::initialize()
{

};


double Korali::Variable::Default::getDensity(double x)
{
 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};

double Korali::Variable::Default::getLogDensity(double x)
{
 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};

double Korali::Variable::Default::getRandomNumber()
{
 fprintf(stderr, "[Korali] Error: Problem requires that variable '%s' to have a defined distribution.\n", _name.c_str());
 exit(-1);
 return 0.0;
};

void Korali::Variable::Default::printDetails()
{
 return;
};

nlohmann::json Korali::Variable::Default::getConfiguration()
{
 auto js = nlohmann::json();

 js["Name"] = _name;
 js["Distribution"] = "Default";

 return js;
}

void Korali::Variable::Default::setConfiguration(nlohmann::json& js)
{
 _name = consume(js, { "Name" }, KORALI_STRING);
}


/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/
