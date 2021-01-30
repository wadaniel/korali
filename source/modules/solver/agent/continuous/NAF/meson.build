# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_NAF_nFDVz7 = custom_target('gen_NAF_nFDVz7',
    output: ['NAF.cpp', 'NAF.hpp'],
    input: files([
      'NAF._cpp',
      'NAF._hpp',
      'NAF.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_NAF_nFDVz7
korali_source_config += files('NAF.config')