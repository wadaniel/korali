# generated with `../../../../../tools/build/korali_gen_target.py`
gen_Nested_suzieT = custom_target('gen_Nested_suzieT',
    output: ['Nested.cpp', 'Nested.hpp'],
    input: files([
      'Nested._cpp',
      'Nested._hpp',
      'Nested.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_Nested_suzieT
korali_source_config += files('Nested.config')