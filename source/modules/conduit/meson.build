gen_conduit = custom_target('conduit_gen',
    output: ['conduit.cpp', 'conduit.hpp'],
    input: files([
      'conduit._cpp',
      'conduit._hpp',
      'conduit.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)

korali_source += gen_conduit
