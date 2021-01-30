# generated with `../../../../tools/build/korali_gen_target.py`
gen_sampler_HTHISE = custom_target('gen_sampler_HTHISE',
    output: ['sampler.cpp', 'sampler.hpp'],
    input: files([
      'sampler._cpp',
      'sampler._hpp',
      'sampler.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_sampler_HTHISE
korali_source_config += files('sampler.config')