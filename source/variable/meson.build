# FIXME: [fabianw@mavt.ethz.ch; 2021-01-29] special version only for testing
korali_gen_variable = find_program('../../tools/build/build_variables_header.py')

gen_variable = custom_target('gen_variable',
    output: ['variable.hpp'],
    input: [files([
      'variable._hpp',
    ]), korali_source_config],
    command: [korali_gen_variable, '--input', '@INPUT@', '--output', '@OUTPUT@']
)
korali_source += gen_variable
