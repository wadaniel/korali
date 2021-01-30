# generated with `../../../../tools/build/korali_gen_target.py`
gen_multivariate_mxHcf9 = custom_target('gen_multivariate_mxHcf9',
    output: ['multivariate.cpp', 'multivariate.hpp'],
    input: files([
      'multivariate._cpp',
      'multivariate._hpp',
      'multivariate.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_multivariate_mxHcf9
korali_source_config += files('multivariate.config')