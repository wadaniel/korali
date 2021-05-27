module_header = files(['deepSupervisor.hpp'])
module_source = files(['deepSupervisor.cpp'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source

subdir('optimizers')
