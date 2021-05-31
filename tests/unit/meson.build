r = run_command(python, '-m', 'korali.cxx', '--cflags')
korali_cflags = r.stdout().strip().split()

r = run_command(python, '-m', 'korali.cxx', '--libs')
korali_libs = r.stdout().strip().split()

engine_tester = executable('engine_tester',
  files(['engine.cpp']),
  dependencies: [ gtest_main_dep ],
  cpp_args: [ korali_cflags ],
  link_args: [ korali_libs ]
  )
  
test('engine', engine_tester,
  protocol: 'gtest',
  suite: 'unit',
)

conduits_tester = executable('conduits_tester',
  files(['conduits.cpp']),
  dependencies: [ gtest_main_dep ],
  cpp_args: [ korali_cflags ],
  link_args: [ korali_libs ]
  )
  
test('conduits', conduits_tester,
  protocol: 'gtest',
  suite: 'unit',
)

distributions_tester = executable('distributions_tester',
  files(['conduits.cpp']),
  dependencies: [ gtest_main_dep ],
  cpp_args: [ korali_cflags ],
  link_args: [ korali_libs ]
  )
  
test('distributions', distributions_tester,
  protocol: 'gtest',
  suite: 'unit',
)
