# generated with `../../../../tools/build/korali_gen_target.py`
gen_sampling_cUdkJ9 = custom_target('gen_sampling_cUdkJ9',
    output: ['sampling.cpp', 'sampling.hpp'],
    input: files([
      'sampling._cpp',
      'sampling._hpp',
      'sampling.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_sampling_cUdkJ9
korali_source_config += files('sampling.config')