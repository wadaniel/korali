# generated with `../../../../../tools/build/korali_gen_target.py`
gen_normal_cr5pHS = custom_target('gen_normal_cr5pHS',
    output: ['normal.cpp', 'normal.hpp'],
    input: files([
      'normal._cpp',
      'normal._hpp',
      'normal.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_normal_cr5pHS
korali_source_config += files('normal.config')