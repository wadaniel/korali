#ifndef _KORALI_AUXILIARS_LOGGER_HPP_
#define _KORALI_AUXILIARS_LOGGER_HPP_

#include <string>

namespace korali
{

extern size_t __verbosityLevel;
extern FILE* __outputFile;

void setVerbosityLevel(const std::string verbosityLevle);
size_t getVerbosityLevel(const std::string verbosityLevel);
void setConsoleOutputFile(FILE* file);
bool isEnoughVerbosity(const std::string verbosityLevel);

void logData(const std::string verbosityLevel, const char* format, ... );
void logInfo(const std::string verbosityLevel, const char* format, ... );
void logWarning(const std::string verbosityLevel, const char* format, ... );
void logError(const char* format, ... );

}
#endif
