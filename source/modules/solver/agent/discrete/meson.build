# generated with `../../../../../tools/build/korali_gen_target.py`
gen_discrete_i5fs65 = custom_target('gen_discrete_i5fs65',
    output: ['discrete.cpp', 'discrete.hpp'],
    input: files([
      'discrete._cpp',
      'discrete._hpp',
      'discrete.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_discrete_i5fs65
korali_source_config += files('discrete.config')