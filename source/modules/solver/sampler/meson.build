module_header = files(['sampler.hpp'])
module_source = files(['sampler.cpp'])
module_config = files(['sampler.config'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source
korali_config += module_config

subdir('HMC')
subdir('MCMC')
subdir('Nested')
subdir('TMCMC')
