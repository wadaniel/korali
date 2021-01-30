# generated with `../../../../../tools/build/korali_gen_target.py`
gen_continuous_8k7hPK = custom_target('gen_continuous_8k7hPK',
    output: ['continuous.cpp', 'continuous.hpp'],
    input: files([
      'continuous._cpp',
      'continuous._hpp',
      'continuous.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_continuous_8k7hPK
korali_source_config += files('continuous.config')