#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "logger.hpp"

size_t Korali::__verbosityLevel;

void Korali::setVerbosityLevel(const  std::string verbosityLevel)
{
 __verbosityLevel = getVerbosityLevel(verbosityLevel);
}

size_t Korali::getVerbosityLevel(const  std::string verbosityLevel)
{
 if (verbosityLevel == "Silent") return 0;
 if (verbosityLevel == "Minimal") return 1;
 if (verbosityLevel == "Normal") return 2;
 if (verbosityLevel == "Detailed") return 3;
 return 0;
}

bool Korali::isEnoughVerbosity(const std::string verbosityLevel)
{
  size_t messageLevel = getVerbosityLevel(verbosityLevel);

  if (messageLevel <= __verbosityLevel) return true;
  return false;
}

void Korali::logData(const std::string verbosityLevel, const char* format, ... )
{
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, format, ap);

 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);

}

void Korali::logInfo(const std::string verbosityLevel, const char* format, ... )
{
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 std::string newFormat = "[Korali] ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::logWarning(const std::string verbosityLevel, const char* format, ... )
{
 if (isEnoughVerbosity(verbosityLevel) == false) return;

 std::string newFormat = "[Korali] Warning: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stderr, "%s", outstr);
 fflush(stderr);
 free(outstr);
}

void Korali::logError(const char* format, ... )
{
 std::string newFormat = "[Korali] Error: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 fprintf(stderr, "%s", outstr);

 fflush(stderr);
 free(outstr);

 std::abort();
}
