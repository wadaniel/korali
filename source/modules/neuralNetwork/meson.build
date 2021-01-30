# generated with `../../../tools/build/korali_gen_target.py`
gen_neuralNetwork_LVW63B = custom_target('gen_neuralNetwork_LVW63B',
    output: ['neuralNetwork.cpp', 'neuralNetwork.hpp'],
    input: files([
      'neuralNetwork._cpp',
      'neuralNetwork._hpp',
      'neuralNetwork.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_neuralNetwork_LVW63B
korali_source_config += files('neuralNetwork.config')