# generated with `../../../../../tools/build/korali_gen_target.py`
gen_weibull_lhU8bx = custom_target('gen_weibull_lhU8bx',
    output: ['weibull.cpp', 'weibull.hpp'],
    input: files([
      'weibull._cpp',
      'weibull._hpp',
      'weibull.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_weibull_lhU8bx
korali_source_config += files('weibull.config')