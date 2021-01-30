# generated with `../../../../tools/build/korali_gen_target.py`
gen_reinforcementLearning_sY47Jk = custom_target('gen_reinforcementLearning_sY47Jk',
    output: ['reinforcementLearning.cpp', 'reinforcementLearning.hpp'],
    input: files([
      'reinforcementLearning._cpp',
      'reinforcementLearning._hpp',
      'reinforcementLearning.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_reinforcementLearning_sY47Jk
korali_source_config += files('reinforcementLearning.config')