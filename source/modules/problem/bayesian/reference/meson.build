# generated with `../../../../../tools/build/korali_gen_target.py`
gen_reference_tH0nip = custom_target('gen_reference_tH0nip',
    output: ['reference.cpp', 'reference.hpp'],
    input: files([
      'reference._cpp',
      'reference._hpp',
      'reference.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_reference_tH0nip
korali_source_config += files('reference.config')