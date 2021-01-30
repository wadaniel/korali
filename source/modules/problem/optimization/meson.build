# generated with `../../../../tools/build/korali_gen_target.py`
gen_optimization_wbCuK3 = custom_target('gen_optimization_wbCuK3',
    output: ['optimization.cpp', 'optimization.hpp'],
    input: files([
      'optimization._cpp',
      'optimization._hpp',
      'optimization.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_optimization_wbCuK3
korali_source_config += files('optimization.config')