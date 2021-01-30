# generated with `../../../../../../tools/build/korali_gen_target.py`
gen_lstm_mmJimy = custom_target('gen_lstm_mmJimy',
    output: ['lstm.cpp', 'lstm.hpp'],
    input: files([
      'lstm._cpp',
      'lstm._hpp',
      'lstm.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_lstm_mmJimy
korali_source_config += files('lstm.config')