project('Korali', ['cpp', 'c'],
  version: '2.0.0',
  license: 'GPL-3.0-only', # or also later?
  default_options: ['cpp_std=c++17', 'b_ndebug=if-release']
)

pymodule = import('python')
python = pymodule.find_installation('python3', required: true)

# list of Korali dependencies
korali_deps = []
cpp_args = []

# check for MPI
mpi_dep = dependency('mpi', language: 'c', required: false)
user_want_mpi = get_option('mpi')
if user_want_mpi and mpi_dep.found()
  korali_deps += mpi_dep
  cpp_args += '-D_KORALI_USE_MPI'
elif user_want_mpi and not mpi_dep.found()
  message('''Configuration requests MPI but could not find implementation.
    Building without MPI support''')
endif

# check for OpenMP support
openmp_dep = dependency('openmp', required: false)
if openmp_dep.found()
  korali_deps += openmp_dep
endif

# process required korali dependencies
korali_deps += dependency('libco', fallback: ['libco', 'libco_dep'], required: true)
korali_deps += dependency('gsl', fallback: ['gsl', 'gsl_dep'], required: true)
korali_deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep'], required: true)

# process optional korali dependencies
# TODO: [fabianw@mavt.ethz.ch; 2021-01-26] 

# generate sources and headers
generator_script = files('tools/build/build.py')
generator_result = run_command(python, generator_script)
if generator_result.returncode() != 0
  error('Can not generate source files using `tools/build/build.py`')
endif
korali_inc = include_directories('include')

# process korali
korali_install_dir = python.get_install_dir() + 'korali'
install_subdir('include', install_dir: korali_install_dir) # cpp headers
subdir('source')

# # create a pkg-config entry
# pkg_mod = import('pkgconfig')
# pkg_mod.generate(
#   name: 'Korali',
#   filebase: 'korali',
#   description: 'High-Performance Uncertainty Quantification, Optimization, and DRL',
#   libraries: pyton_extension
# )
