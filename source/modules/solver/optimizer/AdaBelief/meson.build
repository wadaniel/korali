# generated with `../../../../../tools/build/korali_gen_target.py`
gen_AdaBelief_0EPGUs = custom_target('gen_AdaBelief_0EPGUs',
    output: ['AdaBelief.cpp', 'AdaBelief.hpp'],
    input: files([
      'AdaBelief._cpp',
      'AdaBelief._hpp',
      'AdaBelief.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_AdaBelief_0EPGUs
korali_source_config += files('AdaBelief.config')