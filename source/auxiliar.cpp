#include "json.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include "auxiliar.hpp"
#include "korali.hpp"

Korali::verbosity Korali::_korali_verbosity;

/************************************************************************/
/*                  Python Binding Declarations                         */
/************************************************************************/

#ifdef _KORALI_USE_PYTHON

#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(libkorali, m) {
 pybind11::class_<Korali::Model>(m, "Model")
  .def("getVariable",      &Korali::Model::getVariable, pybind11::return_value_policy::reference)
  .def("getVariableCount", &Korali::Model::getVariableCount, pybind11::return_value_policy::reference)
  .def("getVariables",     &Korali::Model::getVariables, pybind11::return_value_policy::reference)
  .def("getResults",       &Korali::Model::getResults, pybind11::return_value_policy::reference)
 .def("getSampleId",         &Korali::Model::getSampleId, pybind11::return_value_policy::reference)
  #ifdef _KORALI_USE_MPI
  .def("getCommPointer",   &Korali::Model::getCommPointer)
  #endif
  .def("addResult",        &Korali::Model::addResult, pybind11::return_value_policy::reference);

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
 .def("setModel",      &Korali::Engine::setModel, pybind11::return_value_policy::reference)
 .def("addConstraint", &Korali::Engine::addConstraint, pybind11::return_value_policy::reference)
 .def("loadState",     &Korali::Engine::loadState, pybind11::return_value_policy::reference)
 .def("getResults",    &Korali::Engine::getResults, pybind11::return_value_policy::reference);

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

#endif

void Korali::koraliLogData(const verbosity level, const char* format, ... )
{
 if (_k->_conduit != nullptr) if(! _k->_conduit->isRoot()) return;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, format, ap);

 if (level <= _korali_verbosity)
 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::koraliLog(const verbosity level, const char* format, ... )
{
 if (_k->_conduit != nullptr) if(! _k->_conduit->isRoot()) return;

 std::string newFormat = "[Korali] ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 if (level <= _korali_verbosity)
 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::koraliWarning(const verbosity level, const char* format, ... )
{
 if (_k->_conduit != nullptr) if(! _k->_conduit->isRoot()) return;

 std::string newFormat = "[Korali] Warning: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 if (level <= _korali_verbosity) fprintf(stderr, "%s", outstr);
 fflush(stderr);
 free(outstr);
}

void Korali::koraliError(const char* format, ... )
{
 std::string newFormat = "[Korali] Error: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stderr, "%s", outstr);
 fflush(stderr);
 free(outstr);

 std::abort();
}

bool Korali::isEmpty(nlohmann::json& js)
{
 bool empty = true;

 if (js.is_null()) return true;
 if (js.is_primitive()) return false;

 if (js.is_array())
 {
  for (size_t i = 0; i < js.size(); i++)
  {
   bool elEmpty = isEmpty(js[i]);
   if (elEmpty) js.erase(i--);
   empty = empty && elEmpty;
  }
 }

 if (js.is_object())
 {
  std::vector<std::string> erasedKeys;
  for (auto& el : js.items())
  {
   bool elEmpty = isEmpty(el.value());
   if (elEmpty == true) erasedKeys.push_back(el.key());
   empty = empty && elEmpty;
  }
  for (size_t i = 0; i < erasedKeys.size(); i++) js.erase(erasedKeys[i]);
 }

 return empty;
}

void Korali::eraseValue(nlohmann::json& js, std::string path)
{
 std::vector<std::string> settings = getJsonPath(path);

 nlohmann::json* aux = &js;
 size_t i = 0;
 for (; i < settings.size()-1; i++)
  aux = &aux->at(settings[i]);
 aux->erase(settings[i]);
}

bool Korali::isDefined(nlohmann::json& js, std::string path)
{
 return Korali::isDefined(js, getJsonPath(path));
}

bool Korali::isDefined(nlohmann::json& js, std::vector<std::string> settings)
{
 auto tmp = js;

 for (size_t i = 0; i < settings.size(); i++)
 {
  if (tmp.find(settings[i]) == tmp.end()) return false;
  tmp = tmp[settings[i]];
 }
 return true;
}

std::vector<std::string> Korali::getJsonPath(std::string path)
{
 std::vector<size_t> positions;

 size_t curpos = 0;
 while (curpos != std::string::npos)
 {
  if (curpos > 0) positions.push_back(curpos);
  curpos = path.find("'", curpos + 1);
 }

 if (positions.size() % 2 != 0) koraliError("Incorrect path description: %s\n", path.c_str());

 std::vector<std::string> settings;

 for (size_t i = 0; i < positions.size(); i += 2)
 {
  size_t start = positions[i] + 1;
  size_t length = positions[i+1] - start;
  settings.push_back(path.substr(start, length));
 }

 return settings;
}

bool Korali::isArray(nlohmann::json& js, std::vector<std::string> settings)
{
 if (isDefined(js, settings))
 {
  auto tmp = js;
  for (size_t i = 0; i < settings.size(); i++)  tmp = tmp[settings[i]];
  if (tmp.is_array()) if (tmp.size() > 0) return true;
 }
 return false;
}

nlohmann::json Korali::consume(nlohmann::json& js, std::vector<std::string> settings, jsonType type, std::string def)
{
 bool hasDefault = true;

 if (def == "__NO_DEFAULT") hasDefault = false;

 size_t sz =  settings.size();
 std::string fullOption = "";
 for(size_t i = 0; i < sz; i++) fullOption += ">" + settings[i];

 if (isDefined(js, settings))
 {
  nlohmann::json tmp = js;
  nlohmann::json* ptr = &tmp;
  for (size_t i = 0; i < sz-1; i++) ptr = &((*ptr)[settings[i]]);

  bool isCorrect = false;
  if (type == KORALI_STRING && (*ptr)[settings[sz-1]].is_string()) isCorrect = true;
  if (type == KORALI_NUMBER && (*ptr)[settings[sz-1]].is_number()) isCorrect = true;
  if (type == KORALI_ARRAY && (*ptr)[settings[sz-1]].is_array()) isCorrect = true;
  if (type == KORALI_BOOLEAN)
   {
     if ( (*ptr)[settings[sz-1]].is_boolean() || (*ptr)[settings[sz-1]].is_number() ) isCorrect = true;
     if ( (*ptr)[settings[sz-1]].is_number() )
     {
       int val = (*ptr)[settings[sz-1]];
       nlohmann::json newVal = nlohmann::json(val != 0);
       (*ptr)[settings[sz-1]] = newVal;
     }
   }


  if (isCorrect)
  {
   nlohmann::json ret = (*ptr)[settings[sz-1]];
   ptr->erase(settings[sz-1]);
   js = tmp;
   return ret;
  }
  else
  {
   if (type == KORALI_STRING)  koraliError("Passing non-string value to string-type option: %s.\n", fullOption.c_str());
   if (type == KORALI_NUMBER)  koraliError("Passing non-numeric value to numeric option: %s.\n", fullOption.c_str());
   if (type == KORALI_ARRAY)   koraliError("Passing non-array value to array option: %s.\n", fullOption.c_str());
   if (type == KORALI_BOOLEAN) koraliError("Passing non-boolean and non-numeric value to boolean option: %s.\n", fullOption.c_str());
  }
 }

 if (type == KORALI_ARRAY) return nlohmann::json();

 if (hasDefault == false) koraliError("No value passed for non-default option: %s.\n", fullOption.c_str());

 if (type == KORALI_STRING) def = "\"" + def + "\"";
 return nlohmann::json::parse(def);
}

nlohmann::json Korali::loadJsonFromFile(const char* fileName)
{
 nlohmann::json js;

 FILE *fid = fopen(fileName, "r");
 if (fid != NULL)
 {
   fseek(fid, 0, SEEK_END);
   long fsize = ftell(fid);
   fseek(fid, 0, SEEK_SET);  /* same as rewind(f); */

   char *string = (char*) malloc(fsize + 1);
   fread(string, 1, fsize, fid);
   fclose(fid);

   string[fsize] = '\0';

   js = nlohmann::json::parse(string);

   free(string);
 }
 else
  koraliError("Could not load file: %s.\n", fileName);

 return js;
}

void Korali::saveJsonToFile(const char* fileName, nlohmann::json& js)
{
 FILE *fid = fopen(fileName, "w");
 if (fid != NULL)
 {
   fprintf(fid, "%s", js.dump(1).c_str());
   fclose(fid);
 }
 else
  koraliError("Could not write to file: %s.\n", fileName);

}
