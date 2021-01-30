# generated with `../../../../../tools/build/korali_gen_target.py`
gen_custom_9ffueA = custom_target('gen_custom_9ffueA',
    output: ['custom.cpp', 'custom.hpp'],
    input: files([
      'custom._cpp',
      'custom._hpp',
      'custom.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_custom_9ffueA
korali_source_config += files('custom.config')