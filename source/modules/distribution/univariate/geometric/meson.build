# generated with `../../../../../tools/build/korali_gen_target.py`
gen_geometric_uILuzN = custom_target('gen_geometric_uILuzN',
    output: ['geometric.cpp', 'geometric.hpp'],
    input: files([
      'geometric._cpp',
      'geometric._hpp',
      'geometric.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_geometric_uILuzN
korali_source_config += files('geometric.config')