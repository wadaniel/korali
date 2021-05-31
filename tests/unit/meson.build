pythonEmbed = dependency('python3-embed', required: true)

engine_tester = executable('engine_tester',
  files(['engine.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbed ],
  link_with: [ python_extension ]
  )
  
test('engine', engine_tester,
  protocol: 'gtest',
  suite: 'unit',
)

conduits_tester = executable('conduits_tester',
  files(['conduits.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbed ],
  link_with: [ python_extension ]
  )
  
test('conduits', conduits_tester,
  protocol: 'gtest',
  suite: 'unit',
)

distributions_tester = executable('distributions_tester',
  files(['distributions.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbed ],
  link_with: [ python_extension ]
  )
  
test('distributions', distributions_tester,
  protocol: 'gtest',
  suite: 'unit',
)
