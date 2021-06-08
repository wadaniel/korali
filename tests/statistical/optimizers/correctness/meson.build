e = find_program('./.test-run.py', required: true)
test('optimizers.correctness', e,
      timeout : 2000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
