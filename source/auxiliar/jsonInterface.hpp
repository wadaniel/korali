/** \file
* @brief Contains auxiliar functions for JSON object manipulation
******************************************************************************/

#ifndef _KORALI_AUXILIARS_JSONINTERFACE_HPP_
#define _KORALI_AUXILIARS_JSONINTERFACE_HPP_

#include "auxiliar/json.hpp"
#include "auxiliar/logger.hpp"
#include <string>

/**
* \namespace korali
* @brief The Korali namespace includes all Korali-specific functions, variables, and modules.
*/
namespace korali
{
/**
  * @brief Checks whether the JSON object is empty.
  * @param js The JSON object to check.
  * @return true, if it's empty; false, otherwise.
 */
static bool isEmpty(knlohmann::json &js)
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
    for (auto &el : js.items())
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
static bool isElemental(knlohmann::json &js)
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
  else
    isArray = false;

  return isArray;
}

/**
  * @brief Deletes a value on a given JS given a string containing the full path
  * @param js The JSON object to modify.
  * @param key a list of keys describing the full path to traverse
 */
template <typename T, typename... Key>
static void eraseValue(T &js, const Key &... key)
{
  auto *tmp = &js;
  auto *prv = &js;

  bool result = true;
  decltype(tmp->begin()) it;
  ((result && ((it = tmp->find(key)) == tmp->end() ? (result = false) : (prv = tmp, tmp = &*it, true))), ...);

  const auto *lastKey = (&key, ...);

  if (result == true)
    prv->erase(*lastKey);
  else
  {
    std::string keyString(*lastKey);
    KORALI_LOG_ERROR(" + Could not find key '%s' to erase.\n", keyString.c_str());
  }
}

/**
  * @brief Merges the values of two JSON objects recursively and applying priority.
  * @param dest the JSON object onto which the changes will be made. Values here have priority (are not replaced).
  * @param defaults the JSON object that applies onto the other. Values here have no priority (they will not replace)
*/
static void mergeJson(knlohmann::json &dest, const knlohmann::json &defaults)
{
  if (dest.is_object() == false)
  {
    fprintf(stderr, "Passed JSON A argument is not an object.\n");
    exit(-1);
  }
  if (defaults.is_object() == false)
  {
    fprintf(stderr, "Passed JSON B argument is not an object.\n");
    exit(-1);
  }

  for (auto &x : defaults.items())
  {
    auto k = x.key();
    if (dest.find(k) == dest.end()) // Key not found, copy now.
      dest[k] = defaults[k];
    else // Key found, check type.
      if (dest[k].is_object() && defaults[k].is_object())
      mergeJson(dest[k], defaults[k]); // Both are objects. Recurse within.
  }
}

/**
  * @brief Checks whether a given key is present in the JSON object.
  * @param js The JSON object to check.
  * @param key a list of keys describing the full path to traverse
  * @return true, if the path defined by settings is found; false, otherwise.
*/
template <typename T, typename... Key>
static bool isDefined(T &js, const Key &... key)
{
  auto *tmp = &js;
  bool result = true;
  decltype(tmp->begin()) it;
  ((result && ((it = tmp->find(key)) == tmp->end() ? (result = false) : (tmp = &*it, true))), ...);
  return result;
}

/**
  * @brief Returns a value on a given object given a string containing the full path
  * @param js The source object to read from.
  * @param key a list of keys describing the full path to traverse
  * @return Object of the requested path
 */
template <typename T, typename... Key>
static T getValue(T &js, const Key &... key)
{
  auto *tmp = &js;
  auto *prv = &js;

  bool result = true;
  decltype(tmp->begin()) it;
  ((result && ((it = tmp->find(key)) == tmp->end() ? (result = false) : (prv = tmp, tmp = &*it, true))), ...);

  const auto *lastKey = (&key, ...);

  if (result == true)
  {
    return (*prv)[*lastKey];
  }
  else
  {
    std::string keyString(*lastKey);
    KORALI_LOG_ERROR("Could not find key '%s' to return.\n", keyString.c_str());
  }
}

/**
  * @brief Returns a string out of a list of keys showing
  * @param key a list of keys describing the full path to traverse
  * @return The string with a printed key sequence
 */
template <typename... Key>
static std::string getPath(const Key &... key)
{
  std::string path;

  ((path += std::string("[\"") + std::string(key) + std::string("\"]")), ...);

  return path;
}

/**
  * @brief Loads a JSON object from a file.
  * @param dst The JSON object to overwrite.
  * @param fileName The path to the json file to load and parse.
  * @return true, if file was found; false, otherwise.
*/
static bool loadJsonFromFile(knlohmann::json &dst, const char *fileName)
{
  FILE *fid = fopen(fileName, "r");
  if (fid != NULL)
  {
    fseek(fid, 0, SEEK_END);
    long fsize = ftell(fid);
    fseek(fid, 0, SEEK_SET); /* same as rewind(f); */

    char *string = (char *)malloc(fsize + 1);
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
static int saveJsonToFile(const char *fileName, knlohmann::json &js)
{
  FILE *fid = fopen(fileName, "w");
  if (fid != NULL)
  {
    fprintf(fid, "%s", js.dump(1).c_str());
    fclose(fid);
  }
  else
  {
    return -1;
  }

  return 0;
}

} // namespace korali
#endif // _KORALI_AUXILIARS_JSONINTERFACE_HPP_
