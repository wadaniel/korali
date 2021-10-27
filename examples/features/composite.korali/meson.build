e = find_program('./.test-run-cmaes.py', required: true)
test('features.composite.korali', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
