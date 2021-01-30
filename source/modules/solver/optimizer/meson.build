# generated with `../../../../tools/build/korali_gen_target.py`
gen_optimizer_LwECDA = custom_target('gen_optimizer_LwECDA',
    output: ['optimizer.cpp', 'optimizer.hpp'],
    input: files([
      'optimizer._cpp',
      'optimizer._hpp',
      'optimizer.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_optimizer_LwECDA
korali_source_config += files('optimizer.config')