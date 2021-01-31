#libgp build
project('libgp', 'cpp',
        default_options: ['b_ndebug=if-release'])

eigen3_dep = dependency('eigen3', fallback: ['eigen', 'eigen_dep'], required: true)

inc = include_directories('include')

subdir('include')
subdir('src')

# create dependency object that is used by dependent project
libgp_dep = declare_dependency(
  dependencies: eigen3_dep,
  include_directories: inc,
  link_with: libgp
)