# generated with `../../../../../tools/build/korali_gen_target.py`
gen_continuous_9E6lTC = custom_target('gen_continuous_9E6lTC',
    output: ['continuous.cpp', 'continuous.hpp'],
    input: files([
      'continuous._cpp',
      'continuous._hpp',
      'continuous.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_continuous_9E6lTC
korali_source_config += files('continuous.config')