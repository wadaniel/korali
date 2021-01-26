korali_src = files('engine.cpp')

# create a python package
pyton_extension = python.extension_module('korali', 
  korali_src, 
  include_directories: korali_inc,
  dependencies: [korali_deps, python.dependency()],
  install: true,
  install_dir: korali_install_dir,
)
