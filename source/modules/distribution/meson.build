# generated with `../../../tools/build/korali_gen_target.py`
gen_distribution_prDLQl = custom_target('gen_distribution_prDLQl',
    output: ['distribution.cpp', 'distribution.hpp'],
    input: files([
      'distribution._cpp',
      'distribution._hpp',
      'distribution.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_distribution_prDLQl
korali_source_config += files('distribution.config')