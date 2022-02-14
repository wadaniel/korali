e = find_program('run-vracer.py', required: true)
test('reinforcement.learning.cartpole.continuous.single', e,
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('run-vracer-multienv.py', required: true)
test('reinforcement.learning.cartpole.continuous.multiple', e,
      args: [ '--concurrentWorkers', '2'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
    
e = find_program('run-dvracer.py', required: true)
test('reinforcement.learning.cartpole.discrete.single', e,
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
