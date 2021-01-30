# generated with `../../../../tools/build/korali_gen_target.py`
gen_integration_nOwffe = custom_target('gen_integration_nOwffe',
    output: ['integration.cpp', 'integration.hpp'],
    input: files([
      'integration._cpp',
      'integration._hpp',
      'integration.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_integration_nOwffe
korali_source_config += files('integration.config')