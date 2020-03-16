.. _profiler-tool:

***********************************
Korali Profiler
***********************************

parser.add_argument('--input', help='Json files with profiling information to read.', default='_korali_result/profiling.json', nargs='+', required = False)
parser.add_argument('--test', help='Run without graphics (for testing purpose)', action='store_true', required = False)
parser.add_argument('--tend', help='Indicates the maximum time to report in the timeline', required = False, default = 'undefined')
parser.add_argument('--output', help='Indicates the output file path. If not specified, it prints to screen.', required = False)

Usage
========================

Syntax: :code:`python3 -m korali.profiler [--input PROFILING_FILE] [--output OUTPUT_FILE] [--tend END_TIME] [--test]`

Where:

  - :code:`--input` specifies the profiler file to load. By default: :code:`profiling.json`.
  - :code:`--output` indicates the output file path and type (e.g., eps, png). If not specified, it prints to screen.
  - :code:`--tend` indicates time lapse to print. If not specified, it will print the entire execution.
  - :code:`--test` verifies that the plotter works, without plotting to screen.

Examples
========================

Profiling a Single Experiment
----------------------------------


