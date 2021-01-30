# libco build
project('libco', 'c',
  default_options: ['b_ndebug=if-release']
)

src = files([
  'libco.c',
  ])
inc = include_directories('.')
install_headers('libco.h')
libco = static_library('co', src, 
  include_directories: inc, 
  pic: true, 
  install: true
)

# create dependency object that is used by dependent project
libco_dep = declare_dependency(
  include_directories: inc, 
  link_with: libco
)
