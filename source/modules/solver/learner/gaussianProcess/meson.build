# generated with `../../../../../tools/build/korali_gen_target.py`
gen_gaussianProcess_MbkkLk = custom_target('gen_gaussianProcess_MbkkLk',
    output: ['gaussianProcess.cpp', 'gaussianProcess.hpp'],
    input: files([
      'gaussianProcess._cpp',
      'gaussianProcess._hpp',
      'gaussianProcess.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_gaussianProcess_MbkkLk
korali_source_config += files('gaussianProcess.config')