#include "korali.h"
#include <chrono>
#include <sys/types.h>
#include <sys/stat.h>

Korali::Engine* Korali::_k;

/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#ifdef _KORALI_USE_PYTHON

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m) {
 pybind11::class_<Korali::modelData>(m, "modelData")
  .def(pybind11::init<>())
	.def("getParameter", &Korali::modelData::getParameter, pybind11::return_value_policy::reference)
	.def("getParameterCount", &Korali::modelData::getParameterCount, pybind11::return_value_policy::reference)
	.def("getParameters", &Korali::modelData::getParameters, pybind11::return_value_policy::reference)
	.def("getResults", &Korali::modelData::getResults, pybind11::return_value_policy::reference)
  .def("addResult", &Korali::modelData::addResult, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::Engine>(m, "Engine")
 .def(pybind11::init<const std::function<void(Korali::modelData&)>&>())
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
 .def("run", &Korali::Engine::run);

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
 .def(pybind11::init<>())
 .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
 .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
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
 // Determining result folder name
 _currentState = 0;
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

 js["Seed"] = _seed;

 if (_verbosity == KORALI_SILENT)   js["Verbosity"] = "Silent";
 if (_verbosity == KORALI_MINIMAL)  js["Verbosity"] = "Minimal";
 if (_verbosity == KORALI_NORMAL)   js["Verbosity"] = "Normal";
 if (_verbosity == KORALI_DETAILED) js["Verbosity"] = "Detailed";

 js["Output Frequency"] = _outputFrequency;

 for (size_t i = 0; i < N; i++) js["Parameters"][i] = _parameters[i]->getConfiguration();
 js["Problem"] = _problem->getConfiguration();
 js["Solver"] = _solver->getConfiguration();
 js["Conduit"] = _conduit->getConfiguration();

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

 size_t runNumber = 0;
 bool exists = true;
 while(exists)
 {
  struct stat info;
  if (runNumber > 1000) { printf("[Korali] Error: Too many result files. Backup your previous results and run again.\n"); exit(-1);}
  _resultsDirName = "korali" + std::to_string(runNumber);
  if(stat(_resultsDirName.c_str(), &info) != 0) exists = false;
  else if(info.st_mode & S_IFDIR) runNumber++;
 }

 _outputFrequency = consume(js, { "Output Frequency" }, KORALI_NUMBER, "1");
 _resultsDirName = consume(js, { "Output Directory" }, KORALI_STRING, _resultsDirName);

 // Configure Parameters
 std::vector<Korali::Parameter::Base*> tmp;

 if (isArray(js, { "Parameters" } ))
 for (size_t i = 0; i < js["Parameters"].size(); i++)
 {
  auto dString = consume(js["Parameters"][i], { "Distribution" }, KORALI_STRING);
  bool foundDistribution = false;
  if (dString == "Uniform")     { tmp.push_back(new Korali::Parameter::Uniform(js["Parameters"][i], _seed++));     foundDistribution = true; }
  if (dString == "Gaussian")    { tmp.push_back(new Korali::Parameter::Gaussian(js["Parameters"][i], _seed++));    foundDistribution = true; }
  if (dString == "Gamma")       { tmp.push_back(new Korali::Parameter::Gamma(js["Parameters"][i], _seed++));       foundDistribution = true; }
  if (dString == "Exponential") { tmp.push_back(new Korali::Parameter::Exponential(js["Parameters"][i], _seed++)); foundDistribution = true; }
  if (foundDistribution == false) { fprintf(stderr, "[Korali] Error: Incorrect or missing distribution for parameter %lu.\n", i); exit(-1); }
 }

 if (tmp.size() == 0) { fprintf(stderr, "[Korali] Error: Incorrect or undefined parameters.\n"); exit(-1); }

 _statisticalParameterCount = 0;
 _computationalParameterCount = 0;
 for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_COMPUTATIONAL) { _parameters.push_back(tmp[i]); _computationalParameterCount++; }
 for (size_t i = 0; i < tmp.size(); i++) if (tmp[i]->_type == KORALI_STATISTICAL)   { _parameters.push_back(tmp[i]); _statisticalParameterCount++; };
 N = _parameters.size();

 // Configure Problem
 bool foundProblem = false;
 auto pString =  consume(js, { "Problem", "Objective" }, KORALI_STRING);
 if (pString == "Direct Evaluation") { _problem = new Korali::Problem::Direct(js["Problem"]);     foundProblem = true; }
 if (pString == "Likelihood")        { _problem = new Korali::Problem::Likelihood(js["Problem"]); foundProblem = true; }
 if (pString == "Posterior")         { _problem = new Korali::Problem::Posterior(js["Problem"]);  foundProblem = true; }
 if (foundProblem == false) { fprintf(stderr, "[Korali] Error: Incorrect or undefined Problem."); exit(-1); }

 // Configure Conduit
 std::string conduitString = "Sequential";

 conduitString = consume(js, { "Conduit", "Type" }, KORALI_STRING, conduitString);
 _conduit = NULL;

 if (conduitString == "Sequential")
 {
  _conduit = new Korali::Conduit::Sequential(js["Conduit"]);
 }


 if (conduitString == "UPC++")
 {
  #ifdef _KORALI_USE_UPCXX
   _conduit = new Korali::Conduit::UPCXX(js["Conduit"]);
  #else
   fprintf(stderr, "[Korali] Error: UPC++ conduit is not properly configured.\n");
   fprintf(stderr, "[Korali] Reinstall Korali with the proper configuration to support UPC++.\n");
   exit(-1);
  #endif
 }

 if (conduitString == "Multithread")
 {
  #ifdef _KORALI_USE_MULTITHREAD
   _conduit = new Korali::Conduit::Multithread(js["Conduit"]);
  #else
   fprintf(stderr, "[Korali] Error: Multithread conduit is not properly configured.\n");
   fprintf(stderr, "[Korali] Reinstall Korali with the proper configuration to support pthreads.\n");
   exit(-1);
  #endif
 }

 if (_conduit == NULL)
 {
   fprintf(stderr, "[Korali] Error: Unrecognized or no conduit ('%s') selected.\n", conduitString.c_str());
   exit(-1);
 }

 // Configure Solver
 _solver = NULL;
 auto sString = consume(js, { "Solver", "Method" }, KORALI_STRING);
 if (sString == "CMA-ES") _solver = new Korali::Solver::CMAES(js["Solver"]);
 if (sString == "TMCMC")  _solver = new Korali::Solver::TMCMC(js["Solver"]);
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

void Korali::Engine::run()
{
 _k = this;

 setConfiguration(_js);

 #ifdef _KORALI_USE_PYTHON
  pybind11::gil_scoped_release release; // Releasing Global Lock for Multithreaded execution
 #endif

 // Creating Results directory
 mkdir(_resultsDirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

 // Running Engine
 _conduit->run();

 if (_conduit->isRoot()) printf("[Korali] Results saved to folder: '%s'\n", _resultsDirName.c_str());
}

void Korali::Engine::saveState(std::string fileName)
{
 if (!_conduit->isRoot()) return;

 saveJsonToFile(fileName.c_str(), getConfiguration());
}

void Korali::Engine::saveState()
{
 if (!_conduit->isRoot()) return;

 char fileName[256];

 sprintf(fileName, "./%s/s%05lu.json", _resultsDirName.c_str(), _currentState++);

 saveJsonToFile(fileName, getConfiguration());
}

void Korali::Engine::loadState(std::string fileName)
{
 if (!_conduit->isRoot()) return;

 _js = loadJsonFromFile(fileName.c_str());
}

void Korali::Engine::loadConfig(std::string fileName)
{
 _js = loadJsonFromFile(fileName.c_str());
}

