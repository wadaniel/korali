# generated with `../../../../../tools/build/korali_gen_target.py`
gen_multinomial_7Rh8Kh = custom_target('gen_multinomial_7Rh8Kh',
    output: ['multinomial.cpp', 'multinomial.hpp'],
    input: files([
      'multinomial._cpp',
      'multinomial._hpp',
      'multinomial.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_multinomial_7Rh8Kh
korali_source_config += files('multinomial.config')