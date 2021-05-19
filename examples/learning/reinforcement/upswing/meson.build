e = find_program('./.test-run.py', required: true)

current_dir = meson.current_source_dir()

test('Reinforcement Upswing', e,
      args: [current_dir],
      timeout : 1000,
      suite: 'Learning'
    )
