engine_tester = executable('engine_tester',
  files(['engine.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('engine', engine_tester,
  protocol: 'gtest',
  suite: 'unit'
)

auxiliar_tester = executable('auxiliar_tester',
  files(['auxiliar.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('auxiliar', auxiliar_tester,
  protocol: 'gtest',
  suite: 'unit'
)

sample_tester = executable('sample_tester',
  files(['sample.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps,  pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('sample', sample_tester,
  protocol: 'gtest',
  suite: 'unit'
)

subdir('modules')
