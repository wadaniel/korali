#ifndef _KORALI_AUXILIARS_KORALIJSON_HPP_
#define _KORALI_AUXILIARS_KORALIJSON_HPP_

#include <string>
#include "auxiliar/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

namespace korali
{

class Sample;

class KoraliJson {

 public:

 KoraliJson();

 // JSON-based configuration
 nlohmann::json  _js;

 // Python and Json Configuration Binding Methods
 nlohmann::json*  _opt;

 nlohmann::json& getJson();
 void setJson(nlohmann::json& js);

 pybind11::object getItem(pybind11::object key);
 nlohmann::json& operator[](const std::string& key);
 nlohmann::json& operator[](const unsigned long int& key);
 void setItem(pybind11::object key, pybind11::object val);
 void operator=(const std::function<void(korali::Sample&)> val);
 void operator=(const nlohmann::json& val);

 bool contains(const std::string& key);
 void traverseKey(pybind11::object key);
};

}
#endif // _KORALI_AUXILIARS_KORALIJSON_HPP_
