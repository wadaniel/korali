# generated with `../../../../../tools/build/korali_gen_target.py`
gen_linear_obG00W = custom_target('gen_linear_obG00W',
    output: ['linear.cpp', 'linear.hpp'],
    input: files([
      'linear._cpp',
      'linear._hpp',
      'linear.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_linear_obG00W
korali_source_config += files('linear.config')