# generated with `../../../../../tools/build/korali_gen_target.py`
gen_exponential_wbOG7t = custom_target('gen_exponential_wbOG7t',
    output: ['exponential.cpp', 'exponential.hpp'],
    input: files([
      'exponential._cpp',
      'exponential._hpp',
      'exponential.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_exponential_wbOG7t
korali_source_config += files('exponential.config')