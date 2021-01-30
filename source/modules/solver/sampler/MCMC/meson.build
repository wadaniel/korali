# generated with `../../../../../tools/build/korali_gen_target.py`
gen_MCMC_1qLi4X = custom_target('gen_MCMC_1qLi4X',
    output: ['MCMC.cpp', 'MCMC.hpp'],
    input: files([
      'MCMC._cpp',
      'MCMC._hpp',
      'MCMC.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_MCMC_1qLi4X
korali_source_config += files('MCMC.config')