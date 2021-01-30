# generated with `../../../../../tools/build/korali_gen_target.py`
gen_Rprop_qaeXJe = custom_target('gen_Rprop_qaeXJe',
    output: ['Rprop.cpp', 'Rprop.hpp'],
    input: files([
      'Rprop._cpp',
      'Rprop._hpp',
      'Rprop.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_Rprop_qaeXJe
korali_source_config += files('Rprop.config')