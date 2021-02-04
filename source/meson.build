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
# see: https://mesonbuild.com/Reference-tables.html#compiler-ids
compiler_id = meson.get_compiler('cpp').get_id()
link_args = []
if compiler_id == 'gcc' or compiler_id == 'clang' or compiler_id == 'intel'
  link_args = ['-lgsl', '-lgslcblas']
endif
pyton_extension = python.extension_module(korali_extensions,
  korali_source,
  cpp_args: korali_cpp_args,
  link_args: link_args,
  include_directories: korali_include,
  dependencies: [korali_deps, python.dependency()],
  install: true,
  install_dir: korali_install_dir,
)
