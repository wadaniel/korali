if get_option('mpi4py')

e = find_program('./.test-run.py', required: true)
test('features.running.mpi.python', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

endif