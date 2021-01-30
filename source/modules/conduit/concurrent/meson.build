# generated with `../../../../tools/build/korali_gen_target.py`
gen_concurrent_48ykIC = custom_target('gen_concurrent_48ykIC',
    output: ['concurrent.cpp', 'concurrent.hpp'],
    input: files([
      'concurrent._cpp',
      'concurrent._hpp',
      'concurrent.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_concurrent_48ykIC
korali_source_config += files('concurrent.config')