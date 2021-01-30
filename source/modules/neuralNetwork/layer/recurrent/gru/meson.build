# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_gru_A7ZWf3 = custom_target('gen_gru_A7ZWf3',
    output: ['gru.cpp', 'gru.hpp'],
    input: files([
      'gru._cpp',
      'gru._hpp',
      'gru.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_gru_A7ZWf3
korali_source_config += files('gru.config')