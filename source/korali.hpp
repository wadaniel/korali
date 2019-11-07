#include "engine/engine.hpp"
#include "auxiliar/py2json.hpp"
#include "conduit/MPI/MPI.hpp"
#include "conduit/conduit.hpp"

namespace korali
{
 // Main Thread
 size_t _engineCount;

 // Conduit
 extern korali::Conduit* _conduit;

 // State save/load methods
 void saveProfilingInfo();
 void run(std::vector<korali::Engine> engines);

 // Run configuration
 nlohmann::json _js;

 // Communicator Methods
 long int getMPICommPointer();
}
