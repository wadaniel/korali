e = find_program('./.test-run.py', required: true)

test('learning.reinforcement.cartpole', e,
      timeout : 2000,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression'
    )
