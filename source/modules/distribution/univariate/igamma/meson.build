# generated with `../../../../../tools/build/korali_gen_target.py`
gen_igamma_j5ga5A = custom_target('gen_igamma_j5ga5A',
    output: ['igamma.cpp', 'igamma.hpp'],
    input: files([
      'igamma._cpp',
      'igamma._hpp',
      'igamma.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_igamma_j5ga5A
korali_source_config += files('igamma.config')