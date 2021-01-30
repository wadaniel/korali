# generated with `../../../../../tools/build/korali_gen_target.py`
gen_recurrent_CIRAyL = custom_target('gen_recurrent_CIRAyL',
    output: ['recurrent.cpp', 'recurrent.hpp'],
    input: files([
      'recurrent._cpp',
      'recurrent._hpp',
      'recurrent.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_recurrent_CIRAyL
korali_source_config += files('recurrent.config')