# generated with `../../../../../tools/build/korali_gen_target.py`
gen_TMCMC_fUDuzz = custom_target('gen_TMCMC_fUDuzz',
    output: ['TMCMC.cpp', 'TMCMC.hpp'],
    input: files([
      'TMCMC._cpp',
      'TMCMC._hpp',
      'TMCMC.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_TMCMC_fUDuzz
korali_source_config += files('TMCMC.config')