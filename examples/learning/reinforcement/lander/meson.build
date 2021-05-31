e = find_program('./.test-run.py', required: true)

test('learning.reinforcement.lander', e,
      timeout : 3000,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression'
    )
