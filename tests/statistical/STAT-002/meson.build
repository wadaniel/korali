e = find_program('./.test-run.py', required: true)
test('markov.chain', e,
      timeout : 1000,
      suite: 'statistics',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
