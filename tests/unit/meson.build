e = executable('unit',
  files([
    'engine/engine.cpp'
  ]),
  #include_directories: cubismnova_inc,
  dependencies: [gtest_main_dep],
  )
  
test('unit', e,
  protocol: 'gtest',
  suite: 'unit',
)
