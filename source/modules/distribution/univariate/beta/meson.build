# generated with `../../../../../tools/build/korali_gen_target.py`
gen_beta_gjQwnV = custom_target('gen_beta_gjQwnV',
    output: ['beta.cpp', 'beta.hpp'],
    input: files([
      'beta._cpp',
      'beta._hpp',
      'beta.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_beta_gjQwnV
korali_source_config += files('beta.config')