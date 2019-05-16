#ifndef _KORALI_MODEL_MPI_H_
#define _KORALI_MODEL_MPI_H_

#include "models/base/base.h"
#include <mpi.h>

namespace Korali::Model
{

class MPI : public Base {
 public:

 MPI_Comm _comm;
 MPI_Comm& getComm();

 // Evaluation Methods
 virtual void evaluate(std::vector<double>& parameters, std::vector<double>& results);
 std::function<void(Korali::Model::MPI&)> _method;

 // Team Creation and Distribution
 void initialize(size_t nRanks);
 size_t getTeamCount();
 size_t getRanksPerTeam();

 // Constructor / Destructor
 MPI(std::function<void(Korali::Model::MPI&)> method);
 ~MPI();

 // Serialization Methods
 nlohmann::json getConfiguration() override;
 void setConfiguration(nlohmann::json& js) override;
};

} // namespace Korali

#endif // _KORALI_MODEL_MPI_H_
