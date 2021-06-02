e = find_program('./.test-run-cmaes.py', required: true)
test('bayesian.inference.reference.cmaes', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-mcmc.py', required: true)
test('bayesian.inference.reference.mcmc', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-mtmcmc.py', required: true)
test('bayesian.inference.reference.mtmcmc', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-multinest.py', required: true)
test('bayesian.inference.reference.multinest', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-nested.py', required: true)
test('bayesian.inference.reference.nested', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

e = find_program('./.test-run-tmcmc.py', required: true)
test('bayesian.inference.reference.tmcmc', e,
      timeout : 1000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
