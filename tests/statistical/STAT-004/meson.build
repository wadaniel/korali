e = find_program('./.test-run-hmc.py', required: true)
test('samplers.mean.std.hmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-mcmc.py', required: true)
test('samplers.mean.std.mcmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-multinest.py', required: true)
test('samplers.mean.std.multinest', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-nested.py', required: true)
test('samplers.mean.std.nested', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-rchmc.py', required: true)
test('samplers.mean.std.rchmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-tmcmc.py', required: true)
test('samplers.mean.std.tmcmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

