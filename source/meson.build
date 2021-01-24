# TODO: [fabianw@mavt.ethz.ch; 2021-01-24] collect all sources
korali_src = files('engine.cpp')

# create a python package
pypack = python.extension_module('korali', 
  korali_src, 
  dependencies: [korali_deps, python.dependency()])
