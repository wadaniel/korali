# define code generator object
# FIXME: [fabianw@mavt.ethz.ch; 2021-01-29] special version only for testing
korali_gen = find_program('../tools/build/build_fabian.py')

korali_include = include_directories('.')
korali_source = files([
  'engine.cpp',
  'engine.hpp',
  'korali.hpp',
  ])

korali_source_config = [] # collection of .config files used in code generation
subdir('auxiliar')
subdir('modules')
# subdir_done()
# subdir('sample')

# # must be processed last due to source generation dependencies
# subdir('variable')

# create korali python extension
korali_install_dir = python.get_install_dir() + 'korali'
pyton_extension = python.extension_module('korali', 
  korali_source, 
  cpp_args: korali_cpp_args,
  include_directories: korali_include,
  dependencies: [korali_deps, python.dependency()],
  install: true,
  install_dir: korali_install_dir,
)
