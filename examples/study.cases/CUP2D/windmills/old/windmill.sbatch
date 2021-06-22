#!/bin/bash -l
#SBATCH --job-name="eval_flow_zero_3"
#SBATCH --output=eval_flow_zero_3%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=65
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --account=s929
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu

cd ..
# flow_zero_3
./eval-pair-windmill-vracer.sh flow_zero_3

# test flow_zero
#./eval-pair-windmill-vracer.sh flow_zero

# launch uncontrolled run, using energy environment
# ./batch-pair-windmill-vracer.sh uncontrolled

# test flow_zero_4
#./eval-pair-windmill-vracer.sh flow_zero_4

# put back torque to 1e-3
#./eval-pair-windmill-vracer.sh both_copy_long

# run with torque 1e-4
# ./batch-pair-windmill-vracer.sh flow_zero

# launch with target v = 0.25 and time 30s, both 
# ./batch-pair-windmill-vracer.sh velmore


# eval codes again (this time randomized)
#./eval-pair-windmill-vracer.sh both_copy
#./eval-pair-windmill-vracer.sh both_copy_long
#./eval-pair-windmill-vracer.sh energy_zero_copy


# ./batch-pair-windmill-vracer.sh both_double
# ./batch-pair-windmill-vracer.sh flow_zero

# eval energy_zero 
#./eval-pair-windmill-vracer.sh energy_zero_copy

# longer sim with both
#./batch-pair-windmill-vracer.sh both_double

# eval codes 
# ./eval-pair-windmill-vracer.sh both_copy_long
#./eval-pair-windmill-vracer.sh both_copy

# launched from standard code
#./batch-pair-windmill-vracer.sh flow_zero
#./batch-pair-windmill-vracer.sh both_more
#./batch-pair-windmill-vracer.sh energy_zero


#./batch-pair-windmill-vracer.sh energy_false

#./batch-pair-windmill-vracer.sh both
# reward scaling set to true (both launched for flow apparently)
# ./batch-pair-windmill-vracer.sh flow_true
#./batch-pair-windmill-vracer.sh energy_true

# reward scaling set to false (both launched for flow apparently)
#./batch-pair-windmill-vracer.sh flow_false
#./batch-pair-windmill-vracer.sh energy_false

# torque set to 1e-3, reward scaling = true, (these were actually launched for target point 0.7 0.9), both were launched for energy apparently
#./batch-pair-windmill-vracer.sh f_flow
#./batch-pair-windmill-vracer.sh f_energy


#./eval-pair-windmill-vracer.sh torque3_no_vel_copy

# ./batch-pair-windmill-vracer.sh larger_energy
#./batch-pair-windmill-vracer.sh large_energy
#./batch-pair-windmill-vracer.sh energy
#./batch-pair-windmill-vracer.sh both_no_vel

# try with a max torque of 1e-3
# ./batch-pair-windmill-vracer.sh torque3_no_vel 
#./batch-pair-windmill-vracer.sh torque3_vel 

# all below were done with a max torque of 1e-4, also for combo: 2.5 factor of flow, 5e4 for reward
# ./batch-pair-windmill-vracer.sh combo_no_vel 
# ./batch-pair-windmill-vracer.sh combo_vel
# ./batch-pair-windmill-vracer.sh energy_no_vel
# ./batch-pair-windmill-vracer.sh energy_vel
# 
# ./batch-pair-windmill-vracer.sh nan_should
# ./batch-pair-windmill-vracer.sh mod_torque_reward
# ./batch-pair-windmill-vracer.sh only_energy

# ./batch-pair-windmill-vracer.sh 
# ./eval-pair-windmill-vracer.sh

