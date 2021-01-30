# generated with `../../../../../tools/build/korali_gen_target.py`
gen_truncatedNormal_Kk4Aft = custom_target('gen_truncatedNormal_Kk4Aft',
    output: ['truncatedNormal.cpp', 'truncatedNormal.hpp'],
    input: files([
      'truncatedNormal._cpp',
      'truncatedNormal._hpp',
      'truncatedNormal.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_truncatedNormal_Kk4Aft
korali_source_config += files('truncatedNormal.config')