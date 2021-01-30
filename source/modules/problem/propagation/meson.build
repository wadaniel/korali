# generated with `../../../../tools/build/korali_gen_target.py`
gen_propagation_CMHteQ = custom_target('gen_propagation_CMHteQ',
    output: ['propagation.cpp', 'propagation.hpp'],
    input: files([
      'propagation._cpp',
      'propagation._hpp',
      'propagation.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_propagation_CMHteQ
korali_source_config += files('propagation.config')