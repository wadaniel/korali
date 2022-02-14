e = find_program('run-vracer.py', required: true)
test('reinforcement.learning.mountaincart.squashednormal', e,
      args: [ '--distribution', 'Squashed Normal', '--maxExperiences', '10000'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('reinforcement.learning.mountaincart.clippednormal', e,
      args: [ '--distribution', 'Clipped Normal', '--maxExperiences', '10000'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('reinforcement.learning.mountaincart.truncatednormal', e,
      args: [ '--distribution', 'Truncated Normal', '--maxExperiences', '10000'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
