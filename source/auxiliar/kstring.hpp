/** \file
* @brief Auxiliary library for Korali's essential string operations.
**************************************************************************************/

#ifndef _KORALI_AUXILIARS_STRING_HPP_
#define _KORALI_AUXILIARS_STRING_HPP_

#include <string>

namespace korali
{
/**
* @brief Generates lower case string of provided string
* @param input Input string
* @return The lower case varient of the string
*/
extern std::string toLower(const std::string &input);

/**
* @brief Generates upper case string of provided string
* @param input Input string
* @return The upper case variant of the string
*/
extern std::string toUpper(const std::string &input);

/**
* @brief Performs a case-insensitive comparison between strings
* @param a First input string
* @param b Second input string
* @return True if they are a match, false otherwise.
*/
extern bool iCompare(const std::string &a, const std::string &b);

} // namespace korali

#endif // _KORALI_AUXILIARS_STRING_HPP_
