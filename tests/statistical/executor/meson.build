e = find_program('./run-executor-1.py', required: true)
test('samplers.correctness.executor-1', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

