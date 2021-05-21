e = find_program('./.test-run.py', required: true)
test('optimizers.termination', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
