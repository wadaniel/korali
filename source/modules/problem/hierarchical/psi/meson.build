# generated with `../../../../../tools/build/korali_gen_target.py`
gen_psi_1KjqMG = custom_target('gen_psi_1KjqMG',
    output: ['psi.cpp', 'psi.hpp'],
    input: files([
      'psi._cpp',
      'psi._hpp',
      'psi.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_psi_1KjqMG
korali_source_config += files('psi.config')