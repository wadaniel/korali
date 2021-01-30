# generated with `../../../../tools/build/korali_gen_target.py`
gen_SAEM_Hg4ROy = custom_target('gen_SAEM_Hg4ROy',
    output: ['SAEM.cpp', 'SAEM.hpp'],
    input: files([
      'SAEM._cpp',
      'SAEM._hpp',
      'SAEM.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_SAEM_Hg4ROy
korali_source_config += files('SAEM.config')