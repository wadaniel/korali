project_language = 'cpp'
project('Korali', project_language,
  version:
  '2.0.0',
  default_options: ['cpp_std=c++17', 'b_ndebug=if-release']
)

# list of Korali dependencies
korali_deps = []

if get_option('mpi')
  korali_deps += dependency('mpi', language: project_language)
  "-D_KORALI_USE_MPI"
endif

# generate sources
python_exe = find_program('python3')
generator_script = files('source/build.py')
generator_result = run_command(python_exe, generator_script)
if generator_result.returncode() != 0
  error('Can not generate source files using `source/build.py`')
endif
install_subdir('include', install_dir: 'prefix')
korali_include = include_directories('include')
korali_deps += declare_dependency(include_directories: korali_include)

# if get_option('openmp')
#   deps += dependency('openmp')
# endif

# subdir('source')

# Need to be discoverable when installed
pkg_mod = import('pkgconfig')
pkg_mod.generate(
  name: 'Korali',
  filebase: 'korali',
  description: 'High-Performance Uncertainty Quantification, Optimization, and DRL',
)

