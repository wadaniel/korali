pythonEmbedDep = dependency('python3-embed', required: false)

optimizers_tester = executable('optimizers_tester',
  files(['optimizers.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('modules.solver.optimizers', optimizers_tester,
  protocol: 'gtest',
  suite: 'unit',
)

samplers_tester = executable('samplers_tester',
  files(['samplers.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('modules.solver.samplers', samplers_tester,
  protocol: 'gtest',
  suite: 'unit',
)

