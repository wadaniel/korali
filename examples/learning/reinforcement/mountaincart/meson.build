e = find_program('run-vracer.py', required: true)
test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Squashed Normal', '--maxExperiences', '1e4'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Clipped Normal', '--maxExperiences', '1e4'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Truncated Normal', '--maxExperiences', '1e4'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
