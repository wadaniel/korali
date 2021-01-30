# generated with `../../../../../tools/build/korali_gen_target.py`
gen_gamma_qVnBK4 = custom_target('gen_gamma_qVnBK4',
    output: ['gamma.cpp', 'gamma.hpp'],
    input: files([
      'gamma._cpp',
      'gamma._hpp',
      'gamma.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_gamma_qVnBK4
korali_source_config += files('gamma.config')