# generated with `../../../../../tools/build/korali_gen_target.py`
gen_CMAES_8HCGmd = custom_target('gen_CMAES_8HCGmd',
    output: ['CMAES.cpp', 'CMAES.hpp'],
    input: files([
      'CMAES._cpp',
      'CMAES._hpp',
      'CMAES.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_CMAES_8HCGmd
korali_source_config += files('CMAES.config')