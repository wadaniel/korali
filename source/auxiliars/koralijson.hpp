#ifndef _KORALI_AUXILIARS_KORALIJSON_HPP_
#define _KORALI_AUXILIARS_KORALIJSON_HPP_

#include "auxiliars/json.hpp"
#include "modules/sample/sample.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <type_traits>
#include <vector>
#include <iostream>

namespace Korali
{

#pragma GCC diagnostic ignored "-Wunused-function"

enum jsonType { KORALI_STRING, KORALI_NUMBER, KORALI_ARRAY, KORALI_BOOLEAN};

std::vector<std::string> getJsonPath(std::string path);
bool isEmpty(nlohmann::json& js);
void eraseValue(nlohmann::json& js, std::string path);
bool isDefined(nlohmann::json& js, std::vector<std::string> settings);
bool isDefined(nlohmann::json& js, std::string path);
bool isArray(nlohmann::json& js, std::vector<std::string> settings);
nlohmann::json loadJsonFromFile(const char* fileName);
void saveJsonToFile(const char* fileName, nlohmann::json& js);

class KoraliJsonWrapper
{
 public:
  nlohmann::json* _js;
  std::string _key;

  double getValue()
  {
   double val = 0.0;
   if (_js->is_number()) val = *_js;
   else { fprintf(stderr, "[Korali] Error: Attempted getValue() on non-numeric field: %s.\n", _key.c_str());  exit(-1); }
   return val;
  }

  bool getBoolean()
  {
   bool tf = false;
   if (_js->is_boolean()) tf = *_js;
   else { fprintf(stderr, "[Korali] Error: Attempted getBoolean() on non-boolean field: %s.\n", _key.c_str());  exit(-1); }
   return tf;
  }

  std::string getString()
  {
   std::string str;
   if (_js->is_string()) str = *_js;
   else { fprintf(stderr, "[Korali] Error: Attempted getString() on non-string field: %s.\n", _key.c_str());  exit(-1); }
   return str;
  }

  std::vector<double> getArray()
  {
   std::vector<double> vec;
   if (_js->is_array() && (*_js)[0].is_number()) vec = _js->get<std::vector<double>>();
   else { fprintf(stderr, "[Korali] Error: Attempted getArray() on non-array field or non-numeric array: %s.\n", _key.c_str());  exit(-1); }
   return vec;
  }

  KoraliJsonWrapper& getItem(const std::string& key)                   { _key = key; _js = &((*_js)[key]); return *this;}
  KoraliJsonWrapper& getItem(const unsigned long int& key)             { _key = key; _js = &((*_js)[key]); return *this;}
  KoraliJsonWrapper& operator[](const std::string& key)                { _key = key; _js = &((*_js)[key]); return *this;}
  KoraliJsonWrapper& operator[](const unsigned long int& key)          { _key = key; _js = &((*_js)[key]); return *this;}

  void setItem(const std::string& key, const std::function<void(Korali::Sample&)> val);
  void setItem(const std::string& key, const std::string& val)         { _key = key; (*_js)[key] = val; }
  void setItem(const std::string& key, const double& val)              { _key = key; (*_js)[key] = val; }
  void setItem(const std::string& key, const int& val)                 { _key = key; (*_js)[key] = val; }
  void setItem(const std::string& key, const std::vector<double>& val) { _key = key; (*_js)[key] = val; }
  void setItem(const int& key, const std::string& val)                 { _key = key; (*_js)[key] = val; }
  void setItem(const int& key, const double& val)                      { _key = key; (*_js)[key] = val; }
  void setItem(const int& key, const int& val)                         { _key = key; (*_js)[key] = val; }

  void operator=(const std::function<void(Korali::Sample&)> val);
  void operator=(const std::string& val)         { *_js = val; }
  void operator=(const double& val)              { *_js = val; }
  void operator=(const int& val)                 { *_js = val; }
  void operator=(const std::vector<double>& val) { *_js = val; }
};

}
#endif // _KORALI_AUXILIARS_KORALIJSON_HPP_
