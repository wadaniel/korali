module_header = files(['problem.hpp'])
module_source = files(['problem.cpp'])
module_config = files(['problem.config'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source
korali_config += module_config

subdir('bayesian')
subdir('hierarchical')
subdir('integration')
subdir('optimization')
subdir('propagation')
subdir('reinforcementLearning')
subdir('sampling')
subdir('supervisedLearning')
