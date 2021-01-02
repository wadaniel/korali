//  Korali Environment for Nek5000
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.

#include "environment.hpp"
#include "stdio.h"
#include "unistd.h"
#include <dlfcn.h>

void runEnvironment(korali::Sample &s)
{
  // Loading the agent as dynamic library because Nek5000 has non-reentrant code
  // that needs to be wiped out in between runs.
  void *agent = dlopen("./libagent.so", RTLD_LAZY);
  if (!agent)
  {
    fprintf(stderr, "[Korali] Error loading libagent.so (%s).\n", dlerror());
    exit(1);
  }

  // Getting sample ID to create working environment
  size_t sampleId = s["Sample Id"];

  // Creating work environment
  char envdir[1024];
  sprintf(envdir, "_work%lu", sampleId);

  char command[1024];
  sprintf(command, "rm -rf %s", envdir);
  system(command);
  sprintf(command, "mkdir %s", envdir);
  system(command);
  sprintf(command, "cp _config/* %s", envdir);
  system(command);
  chdir(envdir);

  // Storing sample pointer
  auto sample = (korali::Sample **)dlsym(agent, "sample");
  *sample = &s;

  // Initializing environment
  auto comm = MPI_COMM_WORLD;
  void (*nek_init)(MPI_Comm *);
  *(void **)(&nek_init) = dlsym(agent, "nek_init_");
  nek_init(&comm); // When running with MPI, this should be the MPI team

  // Running environment
  void (*nek_solve)();
  *(void **)(&nek_solve) = dlsym(agent, "nek_solve_");
  nek_solve();

  // Cleaning Environment
  void (*nek_end)();
  *(void **)(&nek_end) = dlsym(agent, "nek_end_");
  nek_end();

  // Closing agent dynamic library
  dlclose(agent);
  chdir("..");

  // Setting termination status
  s["Termination"] = "Terminal";
}

