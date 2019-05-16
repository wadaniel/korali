#include "korali.h"

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Model::MPI::MPI(std::function<void(Korali::Model::MPI&)> method) : Korali::Model::Base::Base()
{
 _method = method;

 int isInit;
 MPI_Initialized(&isInit);
 if (!isInit) { fprintf(stderr, "[Korali] Error: MPI needs to be initialized before running Korali.\n"); exit(-1); }
}

Korali::Model::MPI::~MPI()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Model::MPI::getConfiguration()
{
 auto js = this->Korali::Model::Base::getConfiguration();

 js["Type"] = "MPI";

 return js;
}

void Korali::Model::MPI::setConfiguration(nlohmann::json& js)
{
 this->Korali::Model::Base::setConfiguration(js);
}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/


void Korali::Model::MPI::evaluate(std::vector<double>& parameters, std::vector<double>& results)
{
 Korali::Model::MPI evaluator = *this;
 evaluator._self = &evaluator;
 evaluator._comm = MPI_COMM_WORLD;
 evaluator._parameters = parameters;

 _method(evaluator);

 results = evaluator._results;
}

MPI_Comm& Korali::Model::MPI::getComm()
{
 return static_cast<Korali::Model::MPI*>(_self)->_comm;
}

void Korali::Model::MPI::initialize(size_t nRanks)
{
  Base::initialize(nRanks);
}

size_t Korali::Model::MPI::getTeamCount()
{
  return _nRanks;
}

size_t Korali::Model::MPI::getRanksPerTeam()
{
 return 1;
}
