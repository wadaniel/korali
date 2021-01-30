# generated with `../../../../tools/build/korali_gen_target.py`
gen_integrator_ClAigE = custom_target('gen_integrator_ClAigE',
    output: ['integrator.cpp', 'integrator.hpp'],
    input: files([
      'integrator._cpp',
      'integrator._hpp',
      'integrator.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_integrator_ClAigE
korali_source_config += files('integrator.config')