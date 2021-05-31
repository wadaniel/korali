
pythonEmbedLibs = [ ]
pythonEmbedCFlags =  [ ]

pythonEmbedDep = dependency('python3-embed', required: false)
if pythonEmbedDep.found() == false
 r = run_command('python3-config', '--embed', '--cflags')
 pythonEmbedCFlags = r.stdout().strip().split()
 
 r = run_command('python3-config', '--embed', '--ldflags')
 pythonEmbedLibs = r.stdout().strip().split()
endif 


engine_tester = executable('engine_tester',
  files(['engine.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  link_args: [ pythonEmbedLibs ],
  cpp_args: [ pythonEmbedCFlags ]
  )
  
test('engine', engine_tester,
  protocol: 'gtest',
  suite: 'unit',
)

conduits_tester = executable('conduits_tester',
  files(['conduits.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep  ],
  link_with: [ python_extension ],
  link_args: [ pythonEmbedLibs ],
  cpp_args: [ pythonEmbedCFlags ]
  )
  
test('conduits', conduits_tester,
  protocol: 'gtest',
  suite: 'unit',
)

distributions_tester = executable('distributions_tester',
  files(['distributions.cpp']),
  include_directories: korali_include,
  dependencies: [ gtest_main_dep, korali_deps, pythonEmbedDep ],
  link_with: [ python_extension ],
  link_args: [ pythonEmbedLibs ],
  cpp_args: [ pythonEmbedCFlags ]
  )
  
test('distributions', distributions_tester,
  protocol: 'gtest',
  suite: 'unit',
)
