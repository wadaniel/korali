# generated with `../../../../../tools/build/korali_gen_target.py`
gen_output_2qI4yr = custom_target('gen_output_2qI4yr',
    output: ['output.cpp', 'output.hpp'],
    input: files([
      'output._cpp',
      'output._hpp',
      'output.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_output_2qI4yr
korali_source_config += files('output.config')