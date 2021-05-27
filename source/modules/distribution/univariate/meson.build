module_header = files(['univariate.hpp'])
module_source = files(['univariate.cpp'])

install_headers(module_header,
  install_dir: run_command(header_path, [korali_install_headers, meson.current_source_dir()]).stdout().strip()
)

korali_include += include_directories('.')
korali_source += module_header
korali_source += module_source

subdir('beta')
subdir('cauchy')
subdir('exponential')
subdir('gamma')
subdir('geometric')
subdir('igamma')
subdir('laplace')
subdir('logNormal')
subdir('normal')
subdir('truncatedNormal')
subdir('uniform')
subdir('weibull')