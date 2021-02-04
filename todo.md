# Korali core

* [ ] Use new implementation of subprojects/rtnorm
* [X] remove eigen and rtnorm flags
* [X] Provide option to explicitly index the namespace hierarchy (e.g. @startNamespace0)
* [ ] change "" to <> in includes where needed (examples cxx)
* [ ] integrate meson in CircleCI config


# Python module

* [ ] restructure python
* [ ] env script (gitignore)
* [ ] add requirements.txt file


# Third-party

* [X] libco and gsl
* [X] check for more third-party code to move
* [X] rename external to third-party (or subprojects)
* [Χ] optional third-party
* [ ] Fix the `git clean -xdf .` in all `._fetch.sh` scripts. It will fail in a release.


# Documentation

* [ ] doxygen
* [ ] site


# Tests

* [X] return value in all tests
* [ ] write tests in meson
* [ ] configure circleci
* [ ] write separate tests for each file
* [ ] check again the test in running.cxx, running.mpi, reinforcement

# FIXME

* [Χ] pybind is installed with --user and was not found by meson