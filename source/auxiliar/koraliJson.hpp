/** \file
* @brief Contains auxiliar functions for JSON object manipulation and their interface with Korali
******************************************************************************/

#ifndef _KORALI_AUXILIARS_KORALIJSON_HPP_
#define _KORALI_AUXILIARS_KORALIJSON_HPP_

#include <string>
#include "external/json/json.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/functional.h"
#include "pybind11/stl.h"

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{

class Sample;

/**
* \class KoraliJson
* @brief This class encapsulates a JSON object, making it compatible with Korali C++ objects and Pybind11
*/
class KoraliJson {

 public:

 KoraliJson();

 /**
 * @brief Container for the JSON object
 */
 knlohmann::json  _js;

 /**
 * @brief Pointer that stores the current access position of the JSON object.
 *  It advances with getItem, and resets upon setJson or finding a native data type (not a path).
 */
 knlohmann::json*  _opt;

 /**
  * @brief Function to obtain the JSON object.
  * @return A reference to the JSON object.
  */
 knlohmann::json& getJson();

 /**
  * @brief Function to set the JSON object.
  * @param js The input JSON object.
 */
 void setJson(knlohmann::json& js);

 /**
  * @brief Function to obtain a serialized string of the JSON object.
  * @return A serialized string of the JSON object.
  */
 std::string get();

 /**
  * @brief Function to replace the JSON object by deserializing a JSON string
  * @param js The input serialized JSON object.
 */
 void set(const std::string& js);

 /**
  * @brief Gets an item from the JSON object at the current pointer position.
  * @param key A pybind11 object acting as JSON key (number or string).
  * @return A pybind11 object
 */
 pybind11::object getItem(pybind11::object key);

 /**
  * @brief Sets an item on the JSON object at the current pointer position.
  * @param key A pybind11 object acting as JSON key (number or string).
  * @param val The value of the item to set.
 */
 void setItem(pybind11::object key, pybind11::object val);

 /**
  * @brief C++ wrapper for the getItem operator.
  * @param key A C++ string acting as JSON key.
  * @return The referenced JSON object content.
 */
 knlohmann::json& operator[](const std::string& key);

 /**
  * @brief C++ wrapper for the getItem operator.
  * @param key A C++ integer acting as JSON key.
  * @return The referenced JSON object content.
 */
 knlohmann::json& operator[](const unsigned long int& key);

 /**
  * @brief Indicates whether the JSON object contains the given path.
  * @param key key A C++ string acting as JSON key.
  * @return true, if path is found; false, otherwise.
 */
 bool contains(const std::string& key);

 /**
   * @brief Advances the JSON object pointer, given the key
   * @param key A C++ string acting as JSON key.
  */
 void traverseKey(pybind11::object key);
};

/**
* \class JsonInterface
* @brief Provides auxiliar functions for the manipulation of JSON objects
*/
class JsonInterface
{

public:

/**
  * @brief Checks whether the JSON object is empty.
  * @param js The JSON object to check.
  * @return true, if it's empty; false, otherwise.
 */
static bool isEmpty(knlohmann::json& js)
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

/**
  * @brief Checks whether the JSON object is of elemental type (number or string).
  * @param js The JSON object to check.
  * @return true, if it's elemental; false, otherwise.
 */
static bool isElemental(knlohmann::json& js)
{
 if (js.is_number()) return true;
 if (js.is_string()) return true;

 bool isArray = true;

 if (js.is_array())
 {
  for (size_t i = 0; i < js.size(); i++)
  {
   bool isElementArray = false;
   if (js[i].is_array()) isElementArray = isElemental(js[i]);
   if (js[i].is_number()) isElementArray = true;
   if (js[i].is_string()) isElementArray = true;
   isArray = isArray && isElementArray;
  }
 }
 else isArray = false;

 return isArray;
}

/**
  * @brief Decomposes a string of a JSON path (multiple keys) into a set of single keys
  * @param path The string containing multiple keys, e.g., path = "['ABC']['BCD']['EFG']"
  * @return A vector of single strings:  [ 'ABC', 'BCD', 'EFG' ]
 */
static std::vector<std::string> getJsonPath(std::string path)
{
 std::vector<size_t> positions;

 size_t curpos = 0;
 while (curpos != std::string::npos)
 {
  if (curpos > 0) positions.push_back(curpos);
  curpos = path.find("'", curpos + 1);
 }

 if (positions.size() % 2 != 0) { fprintf(stderr, "Incorrect path description: %s\n", path.c_str()); exit(-1); }

 std::vector<std::string> settings;

 for (size_t i = 0; i < positions.size(); i += 2)
 {
  size_t start = positions[i] + 1;
  size_t length = positions[i+1] - start;
  settings.push_back(path.substr(start, length));
 }

 return settings;
}

/**
  * @brief Deletes a value on a given JS given a string containing the full path
  * @param js The JSON object to modify.
  * @param path The string containing the full path, e.g., path = "['ABC']['BCD']['EFG']"
 */
static void eraseValue(knlohmann::json& js, std::string path)
{
 std::vector<std::string> settings = getJsonPath(path);

 knlohmann::json* aux = &js;
 size_t i = 0;
 for (; i < settings.size()-1; i++)
  aux = &aux->at(settings[i]);
 aux->erase(settings[i]);
}

/**
  * @brief Merges the values of two JSON objects recursively and applying priority.
  * @param dest the JSON object onto which the changes will be made. Values here have priority (are not replaced).
  * @param defaults the JSON object that applies onto the other. Values here have no priority (they will not replace)
*/
static void mergeJson(knlohmann::json& dest, const knlohmann::json& defaults)
{
 if (dest.is_object() == false) { fprintf(stderr, "Passed JSON A argument is not an object.\n"); exit(-1); }
 if (defaults.is_object() == false) { fprintf(stderr, "Passed JSON B argument is not an object.\n"); exit(-1); }

 for (auto& x : defaults.items())
 {
  auto k = x.key();
  if (dest.find(k) == dest.end()) // Key not found, copy now.
   dest[k] = defaults[k];
  else                            // Key found, check type.
   if (dest[k].is_object() && defaults[k].is_object()) mergeJson(dest[k], defaults[k]); // Both are objects. Recurse within.
 }
}

/**
  * @brief Checks whether a given key is present in the JSON object.
  * @param js The JSON object to check.
  * @param settings a list of strings containing each a single key.
  * @return true, if the path defined by settings is found; false, otherwise.
*/
static bool isDefined(knlohmann::json& js, std::vector<std::string> settings)
{
 auto tmp = js;

 for (size_t i = 0; i < settings.size(); i++)
 {
  if (tmp.find(settings[i]) == tmp.end()) return false;
  tmp = tmp[settings[i]];
 }
 return true;
}


/**
  * @brief Checks whether a given key is present in the JSON object.
  * @param js The JSON object to check.
  * @param path The string containing the full path, e.g., path = "['ABC']['BCD']['EFG']"
  * @return true, if the path defined by settings is found; false, otherwise.
*/
static bool isDefined(knlohmann::json& js, std::string path)
{
 return isDefined(js, getJsonPath(path));
}

/**
  * @brief Loads a JSON object from a file.
  * @param dst The JSON object to overwrite.
  * @param fileName The path to the json file to load and parse.
  * @return true, if file was found; false, otherwise.
*/
static bool loadJsonFromFile(knlohmann::json& dst, const char* fileName)
{
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

   dst = knlohmann::json::parse(string);

   free(string);

   return true;
 }

 return false;
}

/**
  * @brief Saves a JSON object to a file.
  * @param fileName The path to the file onto which to save the JSON object.
  * @param js The input JSON object.
*/
static void saveJsonToFile(const char* fileName, knlohmann::json& js)
{
 FILE *fid = fopen(fileName, "w");
 if (fid != NULL)
 {
   fprintf(fid, "%s", js.dump(1).c_str());
   fclose(fid);
 }
 else
 { fprintf(stderr, "Could not write to file: %s.\n", fileName); exit(-1); }
}

};

}
#endif // _KORALI_AUXILIARS_KORALIJSON_HPP_
