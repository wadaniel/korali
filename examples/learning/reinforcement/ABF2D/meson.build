e = find_program('./.test-run.py', required: true)

test('learning.reinforcement.ABF2D', e,
      timeout : 1000,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression',
      is_parallel : false
    )
