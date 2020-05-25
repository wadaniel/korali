#include "auxiliar/koraliJson.hpp"
#include "auxiliar/logger.hpp"
#include "auxiliar/py2json.hpp"
#include "sample/sample.hpp"

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

  fprintf(stderr, "Could not recognize Python key format.\n");
  exit(-1);
}

void korali::KoraliJson::setItem(pybind11::object key, pybind11::object val)
{
  traverseKey(key);

  *_opt = val;
  _opt = &_js;
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

knlohmann::json &korali::KoraliJson::operator[](const std::string &key)
{
  return _js[key];
}

knlohmann::json &korali::KoraliJson::operator[](const unsigned long int &key)
{
  return _js[key];
}

knlohmann::json &korali::KoraliJson::getJson()
{
  return _js;
}

void korali::KoraliJson::setJson(knlohmann::json &js)
{
  _js = js;
}

bool korali::KoraliJson::contains(const std::string &key)
{
  if (_js.find(key) == _js.end()) return false;
  return true;
}
