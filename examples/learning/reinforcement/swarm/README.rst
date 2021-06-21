Multi-Agent Reinforcement Learning on Active Particles
=======================================================

Ensemble of N point particles travelling at a fixed speed :math:`|u|=1`.

The reinforcement learning agent :math:`i` that has available as *state* the distance :math:`r_{ij}`, direction vector :math:`\boldsymbol{r}_{ij}`, and angles :math:`\theta_{ij}` to the *M* nearest neighbours :math:`j=1,\dots,M`. The action determines the wished new direction u=(u_x, u_y, u_z) of the particle. The reward is computed as the sum of a pairwise potential between the neihest neighbours

.. math::

   r_t=\sum_{i=0}^{M}V(r)

As example potential is the Lennard-Jones potential as well as its harmonic approximation. The particles orientation is updated by rotating the current orientation by an angle :math:`\alpha` towards the wished new direction. Then it moves by updating the position as :math:`x\rightarrow x+\Delta t u`.

Example
-------

Verbous example with Newton policy  :math:`a=-dV(r)/dr / \|dV(r)/dr\|` and visualisation turned on can be run using 

.. code-block:: bash

  	python main.py --visualize 1 --numIndividuals 10 --numTimesteps 100 --numNearestNeighbours 5