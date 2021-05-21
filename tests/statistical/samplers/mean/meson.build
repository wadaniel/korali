e = find_program('./.test-run-hmc.py', required: true)
test('samplers.mean.hmc', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-mcmc.py', required: true)
test('samplers.mean.mcmc', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-multinest.py', required: true)
test('samplers.mean.multinest', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-nested.py', required: true)
test('samplers.mean.nested', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-rchmc.py', required: true)
test('samplers.mean.rchmc', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-tmcmc.py', required: true)
test('samplers.mean.tmcmc', e,
      timeout : 1000,
      suite: 'statistical',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

