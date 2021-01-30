# generated with `../../../../../tools/build/korali_gen_target.py`
gen_logNormal_EoptYO = custom_target('gen_logNormal_EoptYO',
    output: ['logNormal.cpp', 'logNormal.hpp'],
    input: files([
      'logNormal._cpp',
      'logNormal._hpp',
      'logNormal.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_logNormal_EoptYO
korali_source_config += files('logNormal.config')