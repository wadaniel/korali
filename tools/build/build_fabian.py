#!/usr/bin/env python3
import os
import re
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('--input', nargs=2, type=str, help='Input files', required=True)
    parser.add_argument('--config', type=str, help='Config file', required=True)
    parser.add_argument('--output', nargs=2, type=str, help='Output files')
    # yapf: enable
    return parser.parse_known_args()


def get_output_list(files):
    out = []
    for f in files:
        suffix = os.path.splitext(f)[1].replace('_', '')
        out.append(re.sub(r'\._(hpp|cpp)\s*$', suffix, f))
    return out


def translate(src, conf, dst):
    with open(src, 'r') as content, open(conf, 'r') as config:
        # config is just a dummy
        code = content.read()
    with open(dst, 'w') as out:
        out.write(code)


def main():
    args, _ = parse_args()

    if args.output is None:
        args.output = get_output_list(args.input)

    for src, dst in zip(args.input, args.output):
        translate(src, args.config, dst)


if __name__ == "__main__":
    main()
