#include <stdlib.h>
#include <stdio.h>
#include "logger.hpp"
#include "korali.hpp"

Korali::_korali_verbosity_levels_ Korali::_korali_verbosity;

void Korali::logData(const _korali_verbosity_levels_ level, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, format, ap);

 if (level <= _korali_verbosity)
 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::logInfo(const _korali_verbosity_levels_ level, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;

 std::string newFormat = "[Korali] ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 if (level <= _korali_verbosity)
 fprintf(stdout, "%s", outstr);
 fflush(stdout);
 free(outstr);
}

void Korali::logWarning(const _korali_verbosity_levels_ level, const char* format, ... )
{
 if (Korali::_k->_conduit != nullptr) if(! Korali::_k->_conduit->isRoot()) return;

 std::string newFormat = "[Korali] Warning: ";
 newFormat += format;

 char* outstr = 0;
 va_list ap;
 va_start(ap, format);
 vasprintf(&outstr, newFormat.c_str(), ap);

 if (level <= _korali_verbosity) fprintf(stderr, "%s", outstr);
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
