e = find_program('./.test-run.py', required: true)

test('learning.reinforcement.upswing', e,
      timeout : 3600,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression'
    )
