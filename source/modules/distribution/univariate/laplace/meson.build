# generated with `../../../../../tools/build/korali_gen_target.py`
gen_laplace_U5GCgi = custom_target('gen_laplace_U5GCgi',
    output: ['laplace.cpp', 'laplace.hpp'],
    input: files([
      'laplace._cpp',
      'laplace._hpp',
      'laplace.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_laplace_U5GCgi
korali_source_config += files('laplace.config')