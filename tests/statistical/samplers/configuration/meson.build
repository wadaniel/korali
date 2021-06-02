e = find_program('./run-tmcmc.py', required: true)
test('samplers.configuration.tmcmc', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
