#include "modules/distributions/specific/multinomial/multinomial.hpp"

void Korali::Distribution::Specific::Multinomial::getSelections(std::vector<double>& p, std::vector<unsigned int>& n, int N )
{
 gsl_ran_multinomial(_range, p.size(), N, p.data(), n.data());
}


void Korali::Distribution::Specific::Multinomial::setConfiguration(nlohmann::json& js) 
{
 Korali::Distribution::Specific::Base::setConfiguration(js);
 if(js.count("Configuration Settings") == 1) if (js["Configuration Settings"].empty()) js.erase("Configuration Settings");
 if(js.count("Internal Settings") == 1) if (js["Internal Settings"].empty()) js.erase("Internal Settings");
 if(js.count("Termination Criteria") == 1) if (js["Termination Criteria"].empty()) js.erase("Termination Criteria");
 if(js.count("Variable Configuration") == 1) if (js["Variable Configuration"].empty()) js.erase("Variable Configuration");
 if(js.count("Solver") == 1) if (js["Solver"].empty()) js.erase("Solver");
 if(js.count("Problem") == 1) if (js["Problem"].empty()) js.erase("Problem");
 if(js.count("Conduit") == 1) if (js["Conduit"].empty()) js.erase("Conduit");
 if(js.count("Variables") == 1) for (size_t i = 0; i < js["Variables"].size(); i++) if (js["Variables"][i].empty()) js["Variables"].erase(i);
 if(js.count("Variables") == 1) if (js["Variables"].empty()) js.erase("Variables");
 if(js.empty() == false) Korali::logError("Unrecognized settings for Multinomial Generator (Multinomial): \n%s\n", js.dump(2).c_str());
} 

void Korali::Distribution::Specific::Multinomial::getConfiguration(nlohmann::json& js) 
{

 js["Type"] = "Multinomial";
 Korali::Distribution::Specific::Base::getConfiguration(js);
} 

bool Korali::Distribution::Specific::Multinomial::checkTermination()
{
 bool hasFinished = false;

 return hasFinished;
}

std::string Korali::Distribution::Specific::Multinomial::getType() { return "Multinomial"; }

