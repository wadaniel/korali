/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#include "auxiliars/koralijson.hpp"

#include "modules/engine/model.hpp"
#include "modules/engine/engine.hpp"
#include "modules/conduits/mpi.hpp"

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m)
{
  #ifdef _KORALI_USE_MPI
   m.def("getKoraliMPIComm", &getKoraliMPICommPointer);
  #endif

  pybind11::class_<Korali::Model>(m, "Model")
    .def("__getitem__",     &Korali::Model::operator[], pybind11::return_value_policy::reference)
    .def("getSampleSize",     &Korali::Model::getSampleSize, pybind11::return_value_policy::reference)
    .def("getSampleId",  &Korali::Model::getSampleId, pybind11::return_value_policy::reference)
    .def("setResult", pybind11::overload_cast<const double&>(&Korali::Model::setResult), pybind11::return_value_policy::reference)
    .def("setResult", pybind11::overload_cast<const std::vector<double>&>(&Korali::Model::setResult), pybind11::return_value_policy::reference);

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
   .def("loadState",     &Korali::Engine::loadState, pybind11::return_value_policy::reference);

 pybind11::class_<Korali::KoraliJsonWrapper>(m, "__KoraliJsonWrapper")
   .def(pybind11::init<>())
   .def("getValue", &Korali::KoraliJsonWrapper::getValue)
   .def("getBoolean", &Korali::KoraliJsonWrapper::getBoolean)
   .def("getString", &Korali::KoraliJsonWrapper::getString)
   .def("getArray", &Korali::KoraliJsonWrapper::getArray)
   .def("__getitem__", pybind11::overload_cast<const std::string&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
   .def("__getitem__", pybind11::overload_cast<const unsigned long int&>(&Korali::KoraliJsonWrapper::getItem), pybind11::return_value_policy::reference)
   .def("__setitem__", pybind11::overload_cast<const std::string&, const std::function<void(Korali::Model&)>>(&Korali::KoraliJsonWrapper::setItem), pybind11::return_value_policy::reference)
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
