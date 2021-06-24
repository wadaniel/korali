e = find_program('./test-docs.sh', required: true)
test('docs', e,
      timeout : 1000,
      suite: 'build',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )

e = find_program('./test-style.sh', required: true)
test('style', e,
      timeout : 1000,
      suite: 'build',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )
    