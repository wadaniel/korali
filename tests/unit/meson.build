r = run_command(python, '-m', 'korali.cxx', '--cflags')
korali_cflags = r.stdout().strip().split()

r = run_command(python, '-m', 'korali.cxx', '--libs')
korali_libs = r.stdout().strip().split()

e = executable('unit',
  files(['engine.cpp']),
  dependencies: [ gtest_main_dep ],
  cpp_args: [ korali_cflags ],
  link_args: [ korali_libs ]
  )
  
test('engine', e,
  protocol: 'gtest',
  suite: 'unit',
)
