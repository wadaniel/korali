pythonEmbedDep = dependency('python3-embed', required: false)

fastOpt_tester = executable('fastOpt_tester',
  files(['fastOpt_tester.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  )
  
test('optimizer.fast', fastOpt_tester,
  protocol: 'gtest',
  suite: 'statistical',
)

