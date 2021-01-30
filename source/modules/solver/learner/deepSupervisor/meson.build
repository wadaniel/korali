# generated with `../../../../../tools/build/korali_gen_target.py`
gen_deepSupervisor_L7ITEH = custom_target('gen_deepSupervisor_L7ITEH',
    output: ['deepSupervisor.cpp', 'deepSupervisor.hpp'],
    input: files([
      'deepSupervisor._cpp',
      'deepSupervisor._hpp',
      'deepSupervisor.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_deepSupervisor_L7ITEH
korali_source_config += files('deepSupervisor.config')