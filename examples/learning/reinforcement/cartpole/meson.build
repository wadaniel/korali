e = find_program('./.test-run.py', required: true)

test('learning.reinforcement.cartpole', e,
      args: [current_dir],
      timeout : 1000,
      workdir: meson.current_source_dir(),
      depends: python_extension,
      suite: 'regression'
    )
