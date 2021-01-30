# generated with `../../../tools/build/korali_gen_target.py`
gen_conduit_tOkJoD = custom_target('gen_conduit_tOkJoD',
    output: ['conduit.cpp', 'conduit.hpp'],
    input: files([
      'conduit._cpp',
      'conduit._hpp',
      'conduit.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_conduit_tOkJoD
korali_source_config += files('conduit.config')