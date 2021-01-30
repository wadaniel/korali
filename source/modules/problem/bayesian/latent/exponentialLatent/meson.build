# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_exponentialLatent_KEX9og = custom_target('gen_exponentialLatent_KEX9og',
    output: ['exponentialLatent.cpp', 'exponentialLatent.hpp'],
    input: files([
      'exponentialLatent._cpp',
      'exponentialLatent._hpp',
      'exponentialLatent.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_exponentialLatent_KEX9og
korali_source_config += files('exponentialLatent.config')