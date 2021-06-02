pythonEmbedDep = dependency('python3-embed', required: false)

fcmaes_tester = executable('fcmaes_tester',
  files(['fcmaes_tester.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('statistical.optimizer.fast.cmaes', fcmaes_tester,
  protocol: 'gtest',
  suite: 'statistical',
)

