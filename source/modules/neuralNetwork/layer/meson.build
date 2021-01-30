# generated with `../../../../tools/build/korali_gen_target.py`
gen_layer_m89uZj = custom_target('gen_layer_m89uZj',
    output: ['layer.cpp', 'layer.hpp'],
    input: files([
      'layer._cpp',
      'layer._hpp',
      'layer.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_layer_m89uZj
korali_source_config += files('layer.config')