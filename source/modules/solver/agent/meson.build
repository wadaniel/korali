# generated with `../../../../tools/build/korali_gen_target.py`
gen_agent_UcypyC = custom_target('gen_agent_UcypyC',
    output: ['agent.cpp', 'agent.hpp'],
    input: files([
      'agent._cpp',
      'agent._hpp',
      'agent.config',
    ]),
    command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
)
korali_source += gen_agent_UcypyC
korali_source_config += files('agent.config')