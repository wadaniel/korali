#include "sample/sample.hpp"
#include "engine.hpp"
#include "modules/conduit/conduit.hpp"
#include "modules/experiment/experiment.hpp"
#include "modules/problem/problem.hpp"
#include "modules/solver/solver.hpp"

namespace korali
{
Sample::Sample()
{
  _self = this;
  _state = SampleState::uninitialized;
  _isAllocated = false;
}

void Sample::run(size_t functionPosition)
{
  if (functionPosition >= _functionVector.size())
  {
    fprintf(stderr, "Function ID: %lu not contained in function vector (size: %lu). If you are resuming a previous experiment, you need to re-specify model functions.\n", functionPosition, _functionVector.size());
    exit(-1);
  }
  (*_functionVector[functionPosition])(*_self);
}

void Sample::sampleLauncher()
{
  Engine *engine = _engineStack.top();

  // Getting sample information
  size_t experimentId = KORALI_GET(size_t, (*_self), "Experiment Id");
  auto operation = KORALI_GET(std::string, (*_self), "Operation");
  auto module = KORALI_GET(std::string, (*_self), "Module");

  // Getting experiment pointer
  auto experiment = engine->_experimentVector[experimentId];

  // Running operation
  if ((*_self)["Module"] == "Problem")
    experiment->_problem->runOperation(operation, *_self);

  if ((*_self)["Module"] == "Solver")
    experiment->_solver->runOperation(operation, *_self);
}

knlohmann::json& Sample::globals()
{
 return *_globals;
}

bool Sample::contains(const std::string &key) { return _self->_js.contains(key); }
knlohmann::json &Sample::operator[](const std::string &key) { return _self->_js[key]; }
knlohmann::json &Sample::operator[](const unsigned long int &key) { return _self->_js[key]; }
pybind11::object Sample::getItem(const pybind11::object key) { return _self->_js.getItem(key); }
void Sample::setItem(const pybind11::object key, const pybind11::object val) { _self->_js.setItem(key, val); }

} // namespace korali
