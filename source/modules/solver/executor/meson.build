# generated with `../../../../tools/build/korali_gen_target.py`
gen_executor_GIQdta = custom_target('gen_executor_GIQdta',
    output: ['executor.cpp', 'executor.hpp'],
    input: files([
      'executor._cpp',
      'executor._hpp',
      'executor.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_executor_GIQdta
korali_source_config += files('executor.config')