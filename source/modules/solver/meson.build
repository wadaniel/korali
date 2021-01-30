# generated with `../../../tools/build/korali_gen_target.py`
gen_solver_VXcjLm = custom_target('gen_solver_VXcjLm',
    output: ['solver.cpp', 'solver.hpp'],
    input: files([
      'solver._cpp',
      'solver._hpp',
      'solver.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_solver_VXcjLm
korali_source_config += files('solver.config')