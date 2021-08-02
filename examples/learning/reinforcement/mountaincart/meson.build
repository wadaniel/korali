e = find_program('run-vracer.py', required: true)
test('learning.reinforcement.mountaincart', e,
      timeout : 3600,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )
