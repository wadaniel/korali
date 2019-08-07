/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#include "korali.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m)
{
  m.def("getKoraliMPIComm", &getKoraliMPICommPointer);

  pybind11::class_<Korali::Model::Likelihood>(m, "LikelihoodModel")
    .def("getVariable",     &Korali::Model::Likelihood::getVariable, pybind11::return_value_policy::reference)
    .def("getVariables",     &Korali::Model::Likelihood::getVariables, pybind11::return_value_policy::reference)
    .def("getSampleId",     &Korali::Model::Likelihood::getSampleId, pybind11::return_value_policy::reference)
    .def("setLikelihood",        &Korali::Model::Likelihood::setLikelihood, pybind11::return_value_policy::reference)
    .def("setLogLikelihood",     &Korali::Model::Likelihood::setLogLikelihood, pybind11::return_value_policy::reference);

  pybind11::class_<Korali::Model::Direct>(m, "DirectModel")
    .def("getVariable",     &Korali::Model::Direct::getVariable, pybind11::return_value_policy::reference)
    .def("getVariables",     &Korali::Model::Direct::getVariables, pybind11::return_value_policy::reference)
    .def("getSampleId",     &Korali::Model::Direct::getSampleId, pybind11::return_value_policy::reference)
    .def("setResult",        &Korali::Model::Direct::setResult, pybind11::return_value_policy::reference);

  pybind11::class_<Korali::Model::Reference>(m, "ReferenceModel")
    .def("getVariable",     &Korali::Model::Reference::getVariable, pybind11::return_value_policy::reference)
    .def("getVariables",     &Korali::Model::Reference::getVariables, pybind11::return_value_policy::reference)
    .def("getSampleId",     &Korali::Model::Reference::getSampleId, pybind11::return_value_policy::reference)
    .def("addReferenceEvaluation",        &Korali::Model::Reference::addReferenceEvaluation, pybind11::return_value_policy::reference);

  pybind11::class_<Korali::Model::Constraint>(m, "ConstraintModel")
   .def("getVariable",     &Korali::Model::Constraint::getVariable, pybind11::return_value_policy::reference)
   .def("getVariables",     &Korali::Model::Constraint::getVariables, pybind11::return_value_policy::reference)
   .def("getSampleId",     &Korali::Model::Constraint::getSampleId, pybind11::return_value_policy::reference)
   .def("setConstraintEvaluation",        &Korali::Model::Constraint::setConstraintEvaluation, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::Engine>(m, "Engine")
   .def(pybind11::init<>())
   .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
   .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::Engine::getItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<const std::string&, const std::string&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<const std::string&, const double&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<const std::string&, const int&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<const std::string&, const bool&>(&Korali::Engine::setItem), pybind11::return_value_policy::reference)
   .def("run", &Korali::Engine::run)
   .def("dry", &Korali::Engine::dry)
   .def("setLikelihoodModel", &Korali::Engine::setLikelihoodModel, pybind11::return_value_policy::reference)
   .def("setDirectModel",     &Korali::Engine::setDirectModel, pybind11::return_value_policy::reference)
   .def("setReferenceModel",      &Korali::Engine::setReferenceModel, pybind11::return_value_policy::reference)
   .def("addConstraint", &Korali::Engine::addConstraint, pybind11::return_value_policy::reference)
   .def("loadState",     &Korali::Engine::loadState, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
   .def(pybind11::init<>())
   .def("getValue", &Korali::KoraliJsonWrapper::getValue)
   .def("getBoolean", &Korali::KoraliJsonWrapper::getBoolean)
   .def("getString", &Korali::KoraliJsonWrapper::getString)
   .def("getArray", &Korali::KoraliJsonWrapper::getArray)
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
