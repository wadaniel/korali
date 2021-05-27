module_header = files(['DEA.hpp'])
module_source = files(['DEA.cpp'])
module_config = files(['DEA.config'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source
korali_config += module_config