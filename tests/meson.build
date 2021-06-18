gtest_sub = subproject('gtest')
gtest_dep = gtest_sub.get_variable('gtest_dep')
gtest_main_dep = gtest_sub.get_variable('gtest_main_dep')
gmock_dep = gtest_sub.get_variable('gmock_dep')

# For MPI tests: https://github.com/LLNL/gtest-mpi-listener
# (This is a bug-fixed version from https://github.com/cselab/gtest-mpi-listener)
#gtest_mpi_listener_sub = subproject('gtest-mpi-listener')
#gtest_mpi_listener_dep = gtest_mpi_listener_sub.get_variable('gtest_mpi_listener_dep')

python3_config = find_program('python3-config')
python3_libs = run_command(python3_config, '--ldflags', '--embed').stdout().strip().split()
python3_cflags = run_command(python3_config, '--includes', '--embed').stdout().strip().split()

subdir('python')
subdir('unit')
subdir('statistical')
