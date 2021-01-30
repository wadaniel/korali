# generated with `../../../../../tools/build/korali_gen_target.py`
gen_activation_fx1frq = custom_target('gen_activation_fx1frq',
    output: ['activation.cpp', 'activation.hpp'],
    input: files([
      'activation._cpp',
      'activation._hpp',
      'activation.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_activation_fx1frq
korali_source_config += files('activation.config')