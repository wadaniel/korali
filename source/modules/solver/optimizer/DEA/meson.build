# generated with `../../../../../tools/build/korali_gen_target.py`
gen_DEA_XYoCru = custom_target('gen_DEA_XYoCru',
    output: ['DEA.cpp', 'DEA.hpp'],
    input: files([
      'DEA._cpp',
      'DEA._hpp',
      'DEA.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_DEA_XYoCru
korali_source_config += files('DEA.config')