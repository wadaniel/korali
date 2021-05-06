# define code generator object
korali_gen = find_program('../tools/build/build.py')
# Canonical header path.  Needed for installation as header files are in the
# same place as implementation
header_path = find_program('../tools/build/get_header_directory.py')

configure_file(
  output: 'config.hpp',
  install: true,
  install_dir: korali_install_headers,
  configuration: korali_conf,
)

korali_include = [include_directories('.')]

# If ONEDNN is used, add its path into the includes
if get_option('onednn_path') != ''
  korali_include += get_option('onednn_path') / 'include'
endif

# If CUDNN is used, add its path into the includes
if get_option('cudnn_path') != ''
  korali_include += get_option('cudnn_path') / 'include'
endif

korali_header = files([
  'engine.hpp',
  'korali.hpp',
])
install_headers(korali_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_source = files([
  'engine.cpp',
])
korali_source += korali_header


korali_source_config = [] # collection of .config files used in code generation
subdir('auxiliar')
subdir('modules')
subdir('sample')

# must process last due to dependency on populated korali_source_config list
subdir('variable')

# create korali python extension
python_extension = python.extension_module('lib' + korali,
  korali_source,
  include_directories: korali_include,
  dependencies: [korali_deps, python.dependency()],
  install: true,
  install_dir: korali_install_base,
)