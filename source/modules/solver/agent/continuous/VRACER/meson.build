# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_VRACER_g5xhGA = custom_target('gen_VRACER_g5xhGA',
    output: ['VRACER.cpp', 'VRACER.hpp'],
    input: files([
      'VRACER._cpp',
      'VRACER._hpp',
      'VRACER.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_VRACER_g5xhGA
korali_source_config += files('VRACER.config')