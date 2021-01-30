# generated with `../../../../tools/build/korali_gen_target.py`
gen_hierarchical_rPb2ig = custom_target('gen_hierarchical_rPb2ig',
    output: ['hierarchical.cpp', 'hierarchical.hpp'],
    input: files([
      'hierarchical._cpp',
      'hierarchical._hpp',
      'hierarchical.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_hierarchical_rPb2ig
korali_source_config += files('hierarchical.config')