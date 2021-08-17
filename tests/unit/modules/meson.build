conduits_tester = executable('conduits_tester',
  files(['conduits.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.conduits', conduits_tester,
  protocol: 'gtest',
  suite: 'unit',
)

distributions_tester = executable('distributions_tester',
  files(['distributions.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.distributions', distributions_tester,
  protocol: 'gtest',
  suite: 'unit',
)

problems_tester = executable('problems_tester',
  files(['problems.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.problems', problems_tester,
  protocol: 'gtest',
  suite: 'unit',
)

neuralnetwork_tester = executable('neuralnetwork_tester',
  files(['neuralnetwork.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.neuralnetwork', neuralnetwork_tester,
  protocol: 'gtest',
  suite: 'unit',
)

experiment_tester = executable('experiment_tester',
  files(['experiment.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('modules.experiment', experiment_tester,
  protocol: 'gtest',
  suite: 'unit',
)

subdir('solver')
