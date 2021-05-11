e = find_program('./.test-run.py', required: true)
test('features.running.mpi', e,
      timeout : 1000,
      suite: 'features',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
