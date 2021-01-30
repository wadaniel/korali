# generated with `../../../../../tools/build/korali_gen_target.py`
gen_Adam_aXUBon = custom_target('gen_Adam_aXUBon',
    output: ['Adam.cpp', 'Adam.hpp'],
    input: files([
      'Adam._cpp',
      'Adam._hpp',
      'Adam.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_Adam_aXUBon
korali_source_config += files('Adam.config')