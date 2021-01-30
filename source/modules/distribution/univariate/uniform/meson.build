# generated with `../../../../../tools/build/korali_gen_target.py`
gen_uniform_eM1DRW = custom_target('gen_uniform_eM1DRW',
    output: ['uniform.cpp', 'uniform.hpp'],
    input: files([
      'uniform._cpp',
      'uniform._hpp',
      'uniform.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_uniform_eM1DRW
korali_source_config += files('uniform.config')