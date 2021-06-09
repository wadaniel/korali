pythonEmbedDep = dependency('python3-embed', required: false)

engine_tester = executable('engine_tester',
  files(['engine.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('engine', engine_tester,
  protocol: 'gtest',
  suite: 'unit',
)

auxiliar_tester = executable('auxiliar_tester',
  files(['auxiliar.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('auxiliar', auxiliar_tester,
  protocol: 'gtest',
  suite: 'unit',
)

subdir('modules')
