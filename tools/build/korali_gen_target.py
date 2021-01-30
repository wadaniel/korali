#!/usr/bin/env python3
import sys
import random
import string
import inspect
import argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('name',
                        type=str,
                        help='Build target name (input basename)')
    return parser.parse_known_args()

def main():
    args, _ = parseArgs()
    name = args.name
    chars = string.ascii_letters + string.digits
    target = name + '_' + ''.join(random.choice(chars) for i in range(6))
    with open('meson.build', 'w') as out:
        out.write(
            inspect.cleandoc(f"""
            # generated with `{sys.argv[0]}`
            gen_{target} = custom_target('gen_{target}',
                output: ['{name}.cpp', '{name}.hpp'],
                input: files([
                  '{name}._cpp',
                  '{name}._hpp',
                  '{name}.config',
                ]),
                command: [korali_gen, '--input', '@INPUT0@', '@INPUT1@', '--config', '@INPUT2@', '--output', '@OUTPUT@']
            )
            korali_source += gen_{target}
            korali_source_config += files('{name}.config')
            """))


if __name__ == "__main__":
    main()
