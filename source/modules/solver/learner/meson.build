# generated with `../../../../tools/build/korali_gen_target.py`
gen_learner_xiwfy9 = custom_target('gen_learner_xiwfy9',
    output: ['learner.cpp', 'learner.hpp'],
    input: files([
      'learner._cpp',
      'learner._hpp',
      'learner.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_learner_xiwfy9
korali_source_config += files('learner.config')