Korali
======

High-performance framework for uncertainty quantification, optimization and (inverse) reinforcement learning. Korali's multi-language interface allows the execution of any type of computational model, either sequential or distributed (MPI), C++ or Python, and even pre-compiled/legacy applications. Korali's execution engine enables scalable sampling on large-scale HPC systems. 

Korali provides a simple interface that allows users to easily describe statistical / deep learning problems and choose the algorithms to solve them. The framework can easily be extended to describe new problems or test new experimental algorithms on existing problems.

This repository is a fork of the original project. Here, the RL algorithm was modified for inverse (multi-agent) reinforcement learning applications (IMARL). For implementation details, we refer to the source code located in:

.. code-block:: bash

   ./source/modules/solver/agent/
   ./source/modules/problem/reinforcementLearning

IRL examples for the MuJoCo tasks can be found here:

.. code-block:: bash

    ./study.cases/openAIGym


For more information, read: S. Martin, D. Waelchli, G. Arampatzis, A. E. Economides and P. Karnakov, P. Koumoutsakos, "Korali: Efficient and Scalable Software Framework for Bayesian Uncertainty Quantification and Stochastic Optimization". arXiv 2005.13457. Zurich, Switzerland, March 2021. `[PDF] <https://arxiv.org/abs/2005.13457>`_.

Useful Links
------------

Installation: `https://korali.readthedocs.io/en/v3.0.1/using/install.html`_

Documentation: `https://korali.readthedocs.io/ <https://korali.readthedocs.io/>`_

Website: `https://www.cse-lab.ethz.ch/korali/ <https://www.cse-lab.ethz.ch/korali/>`_ 
