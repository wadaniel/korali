e = find_program('./.run_test.sh', required: true)

test('studycases.openAIGym', e,
      args: [current_dir],
      timeout : 100,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )