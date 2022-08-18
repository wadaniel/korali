#pragma once


/** \file
* @brief Implements a parser for reaction equations
*        based on the implementation by Luca Amoudruz https://github.com/amlucas/SSM
******************************************************************************/

#include <string>
#include <tuple>
#include <vector>

namespace korali
{
    /**
    * @brief Struct of reaction details constructed from reaction equation.
    */
    struct ParsedReactionString
    {
        std::vector<std::string> reactantNames;
        std::vector<int> reactantSCs;
        std::vector<std::string> productNames;
        std::vector<int> productSCs;
        std::vector<bool> isReactantReservoir;
    };

    /**
    * @brief Parses a string and creates a struct of type ParsedReactionString
    * @param s a reaction equation.
    * @return struct containing reaction details.
    */
    ParsedReactionString parseReactionString(std::string s);

} // namespace korali
