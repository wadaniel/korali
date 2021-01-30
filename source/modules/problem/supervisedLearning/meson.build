# generated with `../../../../tools/build/korali_gen_target.py`
gen_supervisedLearning_WaCFJc = custom_target('gen_supervisedLearning_WaCFJc',
    output: ['supervisedLearning.cpp', 'supervisedLearning.hpp'],
    input: files([
      'supervisedLearning._cpp',
      'supervisedLearning._hpp',
      'supervisedLearning.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_supervisedLearning_WaCFJc
korali_source_config += files('supervisedLearning.config')