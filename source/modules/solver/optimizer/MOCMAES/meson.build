# generated with `../../../../../tools/build/korali_gen_target.py`
gen_MOCMAES_9GZBVo = custom_target('gen_MOCMAES_9GZBVo',
    output: ['MOCMAES.cpp', 'MOCMAES.hpp'],
    input: files([
      'MOCMAES._cpp',
      'MOCMAES._hpp',
      'MOCMAES.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_MOCMAES_9GZBVo
korali_source_config += files('MOCMAES.config')