e = find_program('./.test-run-hmc.py', required: true)
test('statistics.samplers.mean.std.hmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-mcmc.py', required: true)
test('statistics.samplers.mean.std.mcmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-multinest.py', required: true)
test('statistics.samplers.mean.std.multinest', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-nested.py', required: true)
test('statistics.samplers.mean.std.nested', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-rhcmc.py', required: true)
test('statistics.samplers.mean.std.rhcmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-tmcmc.py', required: true)
test('statistics.samplers.mean.std.tmcmc', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

