e = find_program('./run-executor-1.py', required: true)
test('executor-1', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./run-executor-2.py', required: true)
test('executor-2', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

