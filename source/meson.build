# define code generator object
korali_gen = find_program('../tools/build/build.py')

korali_include = [include_directories('.')]
korali_source = files([
  'engine.cpp',
  'engine.hpp',
  'korali.hpp',
  ])

korali_source_config = [] # collection of .config files used in code generation
subdir('auxiliar')
subdir('modules')
subdir('sample')

# must process last due to dependency on populated korali_source_config list
subdir('variable')

# create korali python extension
pyton_extension = python.extension_module(korali_extensions,
  korali_source,
  cpp_args: korali_cpp_args,
  include_directories: korali_include,
  dependencies: [korali_deps, python.dependency()],
  install: true,
  install_dir: korali_install_dir,
)
