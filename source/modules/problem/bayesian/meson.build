# generated with `../../../../tools/build/korali_gen_target.py`
gen_bayesian_1XymAp = custom_target('gen_bayesian_1XymAp',
    output: ['bayesian.cpp', 'bayesian.hpp'],
    input: files([
      'bayesian._cpp',
      'bayesian._hpp',
      'bayesian.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_bayesian_1XymAp
korali_source_config += files('bayesian.config')