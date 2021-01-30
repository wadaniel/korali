# generated with `../../../../../tools/build/korali_gen_target.py`
gen_cauchy_RVznGa = custom_target('gen_cauchy_RVznGa',
    output: ['cauchy.cpp', 'cauchy.hpp'],
    input: files([
      'cauchy._cpp',
      'cauchy._hpp',
      'cauchy.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_cauchy_RVznGa
korali_source_config += files('cauchy.config')