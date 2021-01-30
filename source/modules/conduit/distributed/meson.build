# generated with `../../../../tools/build/korali_gen_target.py`
gen_distributed_1AfvIO = custom_target('gen_distributed_1AfvIO',
    output: ['distributed.cpp', 'distributed.hpp'],
    input: files([
      'distributed._cpp',
      'distributed._hpp',
      'distributed.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_distributed_1AfvIO
korali_source_config += files('distributed.config')