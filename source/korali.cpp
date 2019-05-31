#include "korali.h"
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>

Korali::Engine* Korali::_k;

/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#ifdef _KORALI_USE_PYTHON

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m) {
 pybind11::class_<Korali::ModelData>(m, "ModelData")
  .def("getParameter",      &Korali::ModelData::getParameter, pybind11::return_value_policy::reference)
  .def("getParameterCount", &Korali::ModelData::getParameterCount, pybind11::return_value_policy::reference)
  .def("getParameters",     &Korali::ModelData::getParameters, pybind11::return_value_policy::reference)
  .def("getResults",        &Korali::ModelData::getResults, pybind11::return_value_policy::reference)
  #ifdef _KORALI_USE_MPI
  .def("getCommPointer",    &Korali::ModelData::getCommPointer)
  #endif
  .def("addResult",         &Korali::ModelData::addResult, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::Engine>(m, "Engine")
 .def(pybind11::init<>())
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("run", pybind11::overload_cast<std::function<void(Korali::ModelData&)>>(&Korali::Engine::run))
 .def("loadState", pybind11::overload_cast<std::string>(&Korali::Engine::loadState))
 .def("loadConfig", pybind11::overload_cast<std::string>(&Korali::Engine::loadConfig));

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
 .def(pybind11::init<>())
 .def("getValue", &Korali::KoraliJsonWrapper::getValue)
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::vector<double>&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const std::string&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const double&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const int&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const int&, const bool&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference);
}

#endif

/************************************************************************/
/*                  Constructor / Destructor Methods                    */
/************************************************************************/

Korali::Engine::Engine()
{

}

Korali::Engine::~Engine()
{

}

/************************************************************************/
/*                    Configuration Methods                             */
/************************************************************************/

nlohmann::json Korali::Engine::getConfiguration()
{
 auto js = nlohmann::json();

 js["Seed"]      = _seed;

 if (_verbosity == KORALI_SILENT)   js["Verbosity"] = "Silent";
 if (_verbosity == KORALI_MINIMAL)  js["Verbosity"] = "Minimal";
 if (_verbosity == KORALI_NORMAL)   js["Verbosity"] = "Normal";
 if (_verbosity == KORALI_DETAILED) js["Verbosity"] = "Detailed";

 js["Output Frequency"] = _outputFrequency;
 js["Problem"] = _problem->getConfiguration();
 js["Solver"]  = _solver->getConfiguration();
 js["MPI"] = _conduit->getConfiguration();

 return js;
}

void Korali::Engine::setConfiguration(nlohmann::json js)
{
 // Configure Korali Engine

 // Initializing Seed and GSL Random Environment
 _seed = 0;
 FILE *fid = fopen("/dev/random", "rb");
 if (fid != NULL)
 {
  fread(&_seed, 1, sizeof(size_t), fid);
  fclose(fid);
 }
 _seed = consume(js, { "Seed" }, KORALI_NUMBER, std::to_string(_seed));
 gsl_rng_env_setup();

 auto vString = consume(js, { "Verbosity" }, KORALI_STRING, "Normal");
 if (vString == "Silent")   _verbosity = KORALI_SILENT;
 if (vString == "Minimal")  _verbosity = KORALI_MINIMAL;
 if (vString == "Normal")   _verbosity = KORALI_NORMAL;
 if (vString == "Detailed") _verbosity = KORALI_DETAILED;

 _outputFrequency = consume(js, { "Output Frequency" }, KORALI_NUMBER, "1");

 // Configure Problem
 bool foundProblem = false;
 auto pString =  consume(js, { "Problem", "Type" }, KORALI_STRING);
 if (pString == "Direct")   { _problem = new Korali::Problem::Direct(js["Problem"]);   foundProblem = true; }
 if (pString == "Bayesian") { _problem = new Korali::Problem::Bayesian(js["Problem"]); foundProblem = true; }
 if (foundProblem == false) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Problem."); exit(-1); }

 // Configure Conduit

 int rankCount = 1;

 #ifdef _KORALI_USE_MPI
  int isInitialized;
  MPI_Initialized(&isInitialized);
  if (isInitialized == false)  MPI_Init(nullptr, nullptr);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
  if (rankCount > 1) _conduit = new Korali::Conduit::KoraliMPI(js["MPI"]);
 #endif

 if (rankCount == 1) _conduit = new Korali::Conduit::Single(js["MPI"]);

 // Configure Solver
 _solver = NULL;
 auto sString = consume(js, { "Solver", "Method" }, KORALI_STRING);
 if (sString == "CMA-ES")  _solver = new Korali::Solver::CMAES(js["Solver"]);
 if (sString == "CCMA-ES") _solver = new Korali::Solver::CCMAES(js["Solver"]);
 if (sString == "TMCMC")   _solver = new Korali::Solver::TMCMC(js["Solver"]);
 if (_solver == NULL) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Solver."); exit(-1); }

 if (isEmpty(js) == false)
 {
  fprintf(stderr, "[Korali] Error: Unrecognized Settings for Korali:\n");
  fprintf(stderr, "%s\n", js.dump(2).c_str());
  exit(-1);
 }

}

/************************************************************************/
/*                    Functional Methods                                */
/************************************************************************/

void Korali::Engine::run(std::function<void(Korali::ModelData&)> model)
{
 _k = this;

 _model = model;

 setConfiguration(_js);

 // Creating Results directory
 mkdir("_korali_result", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 // Running Engine
 _conduit->run();

 if (_conduit->isRoot()) printf("[Korali] Results saved to folder: '_korali_result'\n");
}

void Korali::Engine::addConstraint(fcon fconstraint)
{
 _fconstraints.push_back(fconstraint);
}


void Korali::Engine::saveState(std::string fileName)
{
 if (!_conduit->isRoot()) return;

 _js = getConfiguration();
 saveJsonToFile(fileName.c_str(), _js);
}

void Korali::Engine::saveState(int fileId)
{
 if (!_conduit->isRoot()) return;

 char fileName[256];

 sprintf(fileName, "./_korali_result/s%05lu.json", fileId++);

 saveState(fileName);
}

void Korali::Engine::loadState(std::string fileName)
{
 // if (!_conduit->isRoot()) return; TODO: produces segfaults, _conduit not init (DW)

 _js = loadJsonFromFile(fileName.c_str());
}

void Korali::Engine::loadConfig(std::string fileName)
{
 _js = loadJsonFromFile(fileName.c_str());
 if (isDefined(_js, {"State"})) _js.erase("State");
}
