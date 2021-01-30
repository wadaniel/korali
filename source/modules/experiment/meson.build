# generated with `../../../tools/build/korali_gen_target.py`
gen_experiment_DL6KGn = custom_target('gen_experiment_DL6KGn',
    output: ['experiment.cpp', 'experiment.hpp'],
    input: files([
      'experiment._cpp',
      'experiment._hpp',
      'experiment.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_experiment_DL6KGn
korali_source_config += files('experiment.config')