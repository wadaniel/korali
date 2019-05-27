#ifndef _KORALI_JSON_H_
#define _KORALI_JSON_H_

#include "json.hpp"
#include <type_traits>
#include <vector>
#include <iostream>

namespace Korali
{

#pragma GCC diagnostic ignored "-Wunused-function"

enum jsonType { KORALI_STRING, KORALI_NUMBER, KORALI_ARRAY, KORALI_BOOLEAN};

class KoraliJsonWrapper
{
 public:
  nlohmann::json* _js;

  KoraliJsonWrapper& getItem(const std::string& key)                   { _js = &((*_js)[key]); return *this;}
  KoraliJsonWrapper& getItem(const unsigned long int& key)             { _js = &((*_js)[key]); return *this;}
  void setItem(const std::string& key, const std::string& val)         { (*_js)[key] = val; }
  void setItem(const std::string& key, const double& val)              { (*_js)[key] = val; }
  void setItem(const std::string& key, const int& val)                 { (*_js)[key] = val; }
  void setItem(const std::string& key, const bool& val)                { (*_js)[key] = val; }
  void setItem(const std::string& key, const std::vector<double>& val) { (*_js)[key] = val; }
  void setItem(const int& key, const std::string& val)                 { (*_js)[key] = val; }
  void setItem(const int& key, const double& val)                      { (*_js)[key] = val; }
  void setItem(const int& key, const int& val)                         { (*_js)[key] = val; }
  void setItem(const int& key, const bool& val)                        { (*_js)[key] = val; }
};

static bool isEmpty(nlohmann::json& js)
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

static bool isDefined(nlohmann::json js, std::vector<std::string> settings)
{
 auto tmp = js;
 for (size_t i = 0; i < settings.size(); i++)
 {
  if (tmp.find(settings[i]) == tmp.end()) return false;
  tmp = tmp[settings[i]];
 }
 return true;
}

static bool isArray(nlohmann::json js, std::vector<std::string> settings)
{
 if (isDefined(js, settings))
 {
  auto tmp = js;
  for (size_t i = 0; i < settings.size(); i++)  tmp = tmp[settings[i]];
  if (tmp.is_array()) if (tmp.size() > 0) return true;
 }
 return false;
}

static nlohmann::json consume(nlohmann::json& js, std::vector<std::string> settings, jsonType type, std::string def = "__NO_DEFAULT" )
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
  if (type == KORALI_BOOLEAN && (*ptr)[settings[sz-1]].is_boolean()) isCorrect = true;

  if (isCorrect)
  {
   nlohmann::json ret = (*ptr)[settings[sz-1]];
   ptr->erase(settings[sz-1]);
   js = tmp;
   return ret;
  }
  else
  {
   if (type == KORALI_STRING)  fprintf(stderr, "[Korali] Error: Passing non-string value to string-type option: %s.\n", fullOption.c_str());
   if (type == KORALI_NUMBER)  fprintf(stderr, "[Korali] Error: Passing non-numeric value to numeric option: %s.\n", fullOption.c_str());
   if (type == KORALI_ARRAY)   fprintf(stderr, "[Korali] Error: Passing non-array value to array option: %s.\n", fullOption.c_str());
   if (type == KORALI_BOOLEAN) fprintf(stderr, "[Korali] Error: Passing non-boolean value to boolean option: %s.\n", fullOption.c_str());
   exit(-1);
  }
 }

 if (hasDefault == false)
 {
  fprintf(stderr, "[Korali] Error: No value passed for non-default option: %s.\n", fullOption.c_str());
  exit(-1);
 }

 if (type == KORALI_STRING) def = "\"" + def + "\"";
 return nlohmann::json::parse(def);
}

}

static nlohmann::json loadJsonFromFile(const char* fileName)
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

   js = nlohmann::json::parse(string);

   free(string);
 }
 else
 {
  fprintf(stderr, "[Korali] Could not load file: %s.\n", fileName);
  exit(-1);
 }

 return js;
}

static void saveJsonToFile(const char* fileName, nlohmann::json js)
{
 FILE *fid = fopen(fileName, "w");
 if (fid != NULL)
 {
   fprintf(fid, "%s", js.dump(1).c_str());
   fclose(fid);
 }
 else
 {
  fprintf(stderr, "[Korali] Could not write to file: %s.\n", fileName);
  exit(-1);
 }
}

#endif // _KORALI_JSON_H_
