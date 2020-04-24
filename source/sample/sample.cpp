#include "sample/sample.hpp"
#include "engine.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"

korali::Sample::Sample()
{
  _self = this;
  _state = SampleState::uninitialized;
  _js["Sample Id"] = 0;
  _isAllocated = false;
}

void korali::Sample::run(size_t functionPosition)
{
 if (functionPosition >= _functionVector.size())
 {
  fprintf(stderr, "Function ID: %lu not contained in function vector (size: %lu). If you are resuming a previous experiment, you need to re-specify model functions.\n", functionPosition, _functionVector.size());
  exit(-1);
 }
 (*_functionVector[functionPosition])(*this);
}

void korali::Sample::update()
{
 korali::Engine* engine = _engineStack.top();
 engine->_currentWorker->update(*_self);
 co_switch(engine->_workerThread);
}

void korali::Sample::sampleLauncher()
{
 korali::Engine* engine = _engineStack.top();
 korali::Sample* sample = engine->_currentSample;

 size_t experimentId = (*sample)["Experiment Id"];
 auto experiment = engine->_experimentVector[experimentId];

 (*sample)["Finished"] = false;
 size_t sampleId = (*sample)["Sample Id"];
 std::string operation = (*sample)["Operation"];

 if ((*sample)["Module"] == "Problem")
  experiment->_problem->runOperation(operation, *sample);

 if ((*sample)["Module"] == "Solver")
  experiment->_solver->runOperation(operation, *sample);

 (*sample)["Finished"] = true;
 engine->_currentWorker->update(*sample);

 co_switch(engine->_workerThread);
}


bool korali::Sample::contains(const std::string& key) { return _self->_js.contains(key); }
knlohmann::json& korali::Sample::operator[](const std::string& key) { return _self->_js[key]; }
knlohmann::json& korali::Sample::operator[](const unsigned long int& key) { return _self->_js[key]; }
pybind11::object korali::Sample::getItem(pybind11::object key) { return _self->_js.getItem(key); }
void korali::Sample::setItem(pybind11::object key, pybind11::object val) { _self->_js.setItem(key, val); }
