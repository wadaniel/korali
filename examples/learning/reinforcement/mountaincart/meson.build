e = find_program('run-vracer.py', required: true)
test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Squashed Normal'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Clipped Normal'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )

test('learning.reinforcement.mountaincart.normal', e,
      args: [ '--distribution', 'Truncated Normal'],
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
