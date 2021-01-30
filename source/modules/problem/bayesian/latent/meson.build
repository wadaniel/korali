# generated with `../../../../../tools/build/korali_gen_target.py`
gen_latent_5ulgRf = custom_target('gen_latent_5ulgRf',
    output: ['latent.cpp', 'latent.hpp'],
    input: files([
      'latent._cpp',
      'latent._hpp',
      'latent.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_latent_5ulgRf
korali_source_config += files('latent.config')