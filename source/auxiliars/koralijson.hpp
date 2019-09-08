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

std::vector<std::string> getJsonPath(std::string path);
bool isEmpty(nlohmann::json& js);
void eraseValue(nlohmann::json& js, std::string path);
bool isDefined(nlohmann::json& js, std::vector<std::string> settings);
bool isDefined(nlohmann::json& js, std::string path);
nlohmann::json loadJsonFromFile(const char* fileName);
void saveJsonToFile(const char* fileName, nlohmann::json& js);

}
#endif // _KORALI_AUXILIARS_KORALIJSON_HPP_
