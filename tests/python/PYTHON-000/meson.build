e = find_program('./test-run.py', required: true)
test('core.cxxtool', e,
      timeout : 100,
      suite: 'core',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )
