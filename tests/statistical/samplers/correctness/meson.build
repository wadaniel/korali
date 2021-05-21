e = find_program('./.test-run.py', required: true)
test('samplers.correctness', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
