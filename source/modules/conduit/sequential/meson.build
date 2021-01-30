# generated with `../../../../tools/build/korali_gen_target.py`
gen_sequential_wpLO4T = custom_target('gen_sequential_wpLO4T',
    output: ['sequential.cpp', 'sequential.hpp'],
    input: files([
      'sequential._cpp',
      'sequential._hpp',
      'sequential.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_sequential_wpLO4T
korali_source_config += files('sequential.config')