Multi-Agent Reinforcement Learning on Active Particles
=======================================================

Ensemble of *N* point particles travelling at a fixed speed |u|=1. The reinforcement learning agent :math:`i` received as *state* the distance :math:`r_{ij}` and angles :math:`\theta_{ij}` to the *M* nearest neighbours :math:`j=1,\dots,M`. The action determines the wished new direction u=(u_x, u_y, u_z) of the particle. The reward is computed as the sum of a pairwise Lennard-Jones potential between the neihest neighbours

.. math::

   r_t=\sum_{i=0}^{M}4\epsilon\left[\left(\frac{\sigma}{r}\right)^{12}-\left(\frac{\sigma}{r}\right)^6\right]

The particles orientation is updated by rotating the current orientation by an angle :math:`\alpha=\tau\Delta t` given a turning rate :math:`\tau` and the time-step :math:`\Delta t`. Then it moves by updating the position as :math:`x\rightarrow x+\Delta t u`.

Example
-------

Verbous example with constant action  :math:`a=(1,0,0)` and visualisation turned on can be run using 

```
python main.py --visualize 1 --numIndividuals 10 --numTimesteps 100 --numNearestNeighbours 5
```