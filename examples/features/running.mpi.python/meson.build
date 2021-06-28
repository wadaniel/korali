e = find_program('./run-cmaes.py', required: true)
test('features.mpi.python', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
