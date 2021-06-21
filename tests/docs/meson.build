e = find_program('./test-docs.sh', required: true)
test('docs.build', e,
      timeout : 1000,
      suite: 'docs',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )
