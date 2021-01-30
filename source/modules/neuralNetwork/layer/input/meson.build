# generated with `../../../../../tools/build/korali_gen_target.py`
gen_input_mY5MvC = custom_target('gen_input_mY5MvC',
    output: ['input.cpp', 'input.hpp'],
    input: files([
      'input._cpp',
      'input._hpp',
      'input.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_input_mY5MvC
korali_source_config += files('input.config')