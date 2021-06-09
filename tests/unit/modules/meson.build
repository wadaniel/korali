pythonEmbedDep = dependency('python3-embed', required: false)

conduits_tester = executable('conduits_tester',
  files(['conduits.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('modules.conduits', conduits_tester,
  protocol: 'gtest',
  suite: 'unit',
)

distributions_tester = executable('distributions_tester',
  files(['distributions.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('modules.distributions', distributions_tester,
  protocol: 'gtest',
  suite: 'unit',
)

subdir('solver')
