# libco build
project('rtnorm', 'cpp',
  default_options: ['b_ndebug=if-release']
)

gsl_dep = dependency('gsl', fallback: ['gsl', 'gsl_dep'], required: true)

src = files([
  'rtnorm.cpp',
  'rtnorm.hpp',
  ])
inc = include_directories('.')
install_headers('rtnorm.hpp')

rtnorm = static_library('rtnorm', src,
  dependencies: gsl_dep,
  include_directories: inc,
  pic: true,
  install: true
)

# create dependency object that is used by dependent project
rtnorm_dep = declare_dependency(
  dependencies: gsl_dep,
  include_directories: inc,
  link_with: rtnorm
)
