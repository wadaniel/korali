#include "auxiliar/koraliJson.hpp"
#include "auxiliar/logger.hpp"
#include "auxiliar/py2json.hpp"
#include "engine/sample/sample.hpp"

korali::KoraliJson::KoraliJson()
{
 _opt = &_js;
}

void korali::KoraliJson::traverseKey(pybind11::object key)
{
 if (pybind11::isinstance<pybind11::str>(key))
 {
  std::string keyStr = key.cast<std::string>();

  _opt = &((*_opt)[keyStr]);
  return;
 }

 if (pybind11::isinstance<pybind11::int_>(key))
  {
   int keyInt = key.cast<int>();
   _opt = &((*_opt)[keyInt]);
   return;
  }

 korali::logError("Could not recognize Python key format.\n");
}

pybind11::object korali::KoraliJson::getItem(pybind11::object key)
{
 traverseKey(key);

 if (JsonInterface::isElemental(*_opt))
 {
  auto tmp = _opt;
  _opt = &_js;
  return *tmp;
 }

 return pybind11::cast(this);
}

void korali::KoraliJson::operator=(const std::function<void(korali::Sample&)> val)
{
 (*_opt) = (std::uint64_t)&val;
 _opt = &_js;
}

nlohmann::json& korali::KoraliJson::operator[](const std::string& key)
{
 return _js[key];
}

nlohmann::json& korali::KoraliJson::operator[](const unsigned long int& key)
{
 return _js[key];
}

void korali::KoraliJson::setItem(pybind11::object key, pybind11::object val)
{
 traverseKey(key);

 *_opt = val;
 _opt = &_js;
}

void korali::KoraliJson::operator=(const nlohmann::json& val)
{
 (*_opt) = val;
 _opt = &_js;
}

nlohmann::json& korali::KoraliJson::getJson()
{
 return _js;
}

void korali::KoraliJson::setJson(nlohmann::json& js)
{
 _js = js;
}

bool korali::KoraliJson::contains(const std::string& key)
{
 if (_js.find(key) == _js.end()) return false;
 return true;
}
