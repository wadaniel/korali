e = find_program('./test-run.py', required: true)
test('python.cxx', e,
      timeout : 100,
      suite: 'python',
      workdir: meson.current_source_dir(),
      depends: python_extension
    )
