fastOpt_tester = executable('fastOpt_tester',
  files(['fastOpt_tester.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, python3_embed, pybind11_dep ],
  link_with: [ python_extension ],
  )
  
test('optimizer.fast', fastOpt_tester,
  protocol: 'gtest',
  timeout : 1000,
  suite: 'statistical',
)

