fastOpt_tester = executable('fastOpt_tester',
  files(['fastOpt_tester.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pybind11_dep ],
  link_with: [ python_extension ],
  link_args: [ python3_libs ],
  cpp_args: [ python3_cflags ]
  )
  
test('optimizer.fast', fastOpt_tester,
  protocol: 'gtest',
  timeout : 1000,
  suite: 'statistical',
)

