project('Korali', ['c', 'cpp'],
  version: '2.0.0',
  license: 'GPL-3.0-only', # or also later?
  default_options: ['cpp_std=c++17', 'b_ndebug=if-release']
)

# python installation
pymodule = import('python')
python = pymodule.find_installation('python3', required: true)

korali = 'korali' # python package name
korali_extensions = 'libkorali' # cpp extensions name
korali_install_dir = python.get_install_dir() / korali

# list of Korali dependencies
korali_deps = []
korali_cpp_args = []

# check for MPI
mpi_dep = dependency('mpi', language: 'c', required: false)
user_want_mpi = get_option('mpi')
if user_want_mpi and mpi_dep.found()
  korali_deps += mpi_dep
  korali_cpp_args += '-D_KORALI_USE_MPI'
elif user_want_mpi and not mpi_dep.found()
  message('''Configuration requests MPI but could not find implementation.
    Building without MPI support''')
endif

# check for OpenMP support
korali_deps += dependency('openmp', required: false)


# process optional korali dependencies
onednn_dep = dependency('onednn', required: false)
user_want_onednn = get_option('onednn')
if user_want_onednn and onednn_dep.found()
  korali_deps += onednn_dep
  korali_cpp_args += '-D_KORALI_USE_ONEDNN'
elif user_want_onednn and not onednn_dep.found()
  onednn_version = '2.0'
  _r = run_command('subprojects/onednn/._fetch.sh', [onednn_version, join_paths('subprojects','onednn')])

  cmake = import('cmake')

  opt_var = cmake.subproject_options()
  opt_var.add_cmake_defines({
      'DNNL_BUILD_EXAMPLES': false,
      'DNNL_BUILD_TESTS': false,
      'DNNL_ENABLE_CONCURRENT_EXEC': true,
      'CMAKE_INSTALL_PREFIX': '@PREFIX@',
      'DNNL_CPU_RUNTIME': 'SEQ',
      'BUILD_SHARED_LIBS': true,
      })

  sub_proj = cmake.subproject('onednn', options: opt_var)

  # message('CMaket targets:\n - ' + '\n - '.join(sub_proj.target_list()))
  oneDNN_dep = sub_proj.dependency('dnnl')
endif

# process required korali dependencies
korali_deps += dependency('eigen3', fallback: ['eigen', 'eigen_dep'], required: true)
korali_deps += dependency('gsl', fallback: ['gsl', 'gsl_dep'], required: true)
korali_deps += dependency('libco', fallback: ['libco', 'libco_dep'], required: true)
korali_deps += dependency('libgp', fallback: ['libgp', 'libgp_dep'], required: true)
korali_deps += dependency('pybind11', fallback: ['pybind11', 'pybind11_dep'], required: true)
korali_deps += dependency('rtnorm', fallback: ['rtnorm', 'rtnorm_dep'], required: true)



# process korali
subdir('source')

# FIXME: [fabianw@mavt.ethz.ch; 2021-01-29] this must be generated in the
# installation stage
subdir_done()
korali_inc = include_directories('include')
install_subdir('include', install_dir: korali_install_dir) # cpp headers

# # create a pkg-config entry
# pkg_mod = import('pkgconfig')
# pkg_mod.generate(
#   name: 'Korali',
#   filebase: 'korali',
#   description: 'High-Performance Uncertainty Quantification, Optimization, and DRL',
#   libraries: pyton_extension
# )
