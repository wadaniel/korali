# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_dVRACER_GGbbCF = custom_target('gen_dVRACER_GGbbCF',
    output: ['dVRACER.cpp', 'dVRACER.hpp'],
    input: files([
      'dVRACER._cpp',
      'dVRACER._hpp',
      'dVRACER.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_dVRACER_GGbbCF
korali_source_config += files('dVRACER.config')