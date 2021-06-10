e = find_program('run-mnist.py', required: true)
test('learning.supervised.mnist', e,
      args: [ '--test' ],
      timeout : 2000,
      suite: 'regression',
      workdir: meson.current_source_dir(),
      depends: python_extension,
      env: nomalloc
    )