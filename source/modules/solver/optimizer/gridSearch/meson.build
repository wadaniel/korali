# generated with `../../../../../tools/build/korali_gen_target.py`
gen_gridSearch_4NFP93 = custom_target('gen_gridSearch_4NFP93',
    output: ['gridSearch.cpp', 'gridSearch.hpp'],
    input: files([
      'gridSearch._cpp',
      'gridSearch._hpp',
      'gridSearch.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_gridSearch_4NFP93
korali_source_config += files('gridSearch.config')