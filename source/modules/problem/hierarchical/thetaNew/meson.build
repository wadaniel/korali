# generated with `../../../../../tools/build/korali_gen_target.py`
gen_thetaNew_SvpUGf = custom_target('gen_thetaNew_SvpUGf',
    output: ['thetaNew.cpp', 'thetaNew.hpp'],
    input: files([
      'thetaNew._cpp',
      'thetaNew._hpp',
      'thetaNew.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_thetaNew_SvpUGf
korali_source_config += files('thetaNew.config')