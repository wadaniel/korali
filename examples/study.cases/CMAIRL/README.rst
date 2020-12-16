Study Case: Inverse Deep RL with the Cartpole
=================================================================================

Learning the reward function, which is the cosinus of the deviation from the target angle

CMA-ES is used to find the true angle. The objective function is the L2-norm of the deviations from observed and generated actions (forces applied to the cartpole).
