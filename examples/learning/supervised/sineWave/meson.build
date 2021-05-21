e = find_program('./.test-run.py', required: true)
test('learning.supervised.sineWave', e,
      timeout : 2000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc,
      is_parallel : false
    )
