# generated with `../../../tools/build/korali_gen_target.py`
gen_problem_PUHzFU = custom_target('gen_problem_PUHzFU',
    output: ['problem.cpp', 'problem.hpp'],
    input: files([
      'problem._cpp',
      'problem._hpp',
      'problem.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_problem_PUHzFU
korali_source_config += files('problem.config')