e = find_program('./.test-run.py', required: true)

current_dir = meson.current_source_dir()

test('Reinforcement ABF2D', e,
      args: [current_dir],
      timeout : 1000,
      suite: 'Learning'
    )
