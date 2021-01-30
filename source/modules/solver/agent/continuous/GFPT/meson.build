# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_GFPT_3fBdt2 = custom_target('gen_GFPT_3fBdt2',
    output: ['GFPT.cpp', 'GFPT.hpp'],
    input: files([
      'GFPT._cpp',
      'GFPT._hpp',
      'GFPT.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_GFPT_3fBdt2
korali_source_config += files('GFPT.config')