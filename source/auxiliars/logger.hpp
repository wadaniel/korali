#ifndef _KORALI_AUXILIARS_LOGGER_HPP_
#define _KORALI_AUXILIARS_LOGGER_HPP_

#include <string>

namespace Korali
{

size_t getVerbosityLevel(std::string verbosityLevel);
bool isEnoughVerbosity(std::string verbosityLevel);

void logData(std::string verbosityLevel, const char* format, ... );
void logInfo(std::string verbosityLevel, const char* format, ... );
void logWarning(std::string verbosityLevel, const char* format, ... );
void logError(const char* format, ... );

}
#endif
