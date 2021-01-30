# generated with `../../../../../tools/build/korali_gen_target.py`
gen_theta_BwN221 = custom_target('gen_theta_BwN221',
    output: ['theta.cpp', 'theta.hpp'],
    input: files([
      'theta._cpp',
      'theta._hpp',
      'theta.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_theta_BwN221
korali_source_config += files('theta.config')