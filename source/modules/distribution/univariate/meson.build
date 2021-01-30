# generated with `../../../../tools/build/korali_gen_target.py`
gen_univariate_WNCiXM = custom_target('gen_univariate_WNCiXM',
    output: ['univariate.cpp', 'univariate.hpp'],
    input: files([
      'univariate._cpp',
      'univariate._hpp',
      'univariate.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_univariate_WNCiXM
korali_source_config += files('univariate.config')