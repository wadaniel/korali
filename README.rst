***********
Korali
***********

High-performance framework for uncertainty quantification, optimization and reinforcement learning.

.. image:: https://circleci.com/gh/cselab/korali.svg?style=shield
    :target: https://circleci.com/gh/cselab/korali
    :alt: Build Status
.. image:: https://readthedocs.org/projects/korali/badge/?version=master
    :target: https://korali.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status
.. image:: https://codecov.io/gh/cselab/korali/branch/prerelease/graph/badge.svg?token=yyY5Ew6T8N
    :target: https://codecov.io/gh/cselab/korali
    :alt: Code Coverage

Description
===========

Korali is a high-performance framework for Bayesian UQ, optimization, and reinforcement learning. Korali's multi-language interface allows the execution of any type of computational model, either sequential or distributed (MPI), C++ or Python, and even pre-compiled/legacy applications. Korali's execution engine enables scalable sampling on large-scale HPC systems. 

Korali provides a simple interface that allows users to easily describe statistical / deep learning problems and choose the algorithms to solve them. The framework can easily be extended to describe new problems or test new experimental algorithms on existing problems.

For more information, read: S. Martin, D. Waelchli, G. Arampatzis, A. E. Economides and P. Karnakov, P. Koumoutsakos, "Korali: Efficient and Scalable Software Framework for Bayesian Uncertainty Quantification and Stochastic Optimization". arXiv 2005.13457. Zurich, Switzerland, March 2021. `[PDF] <https://arxiv.org/abs/2005.13457>`_.

Website
========

Visit: https://www.cse-lab.ethz.ch/korali/ for more information.

Folders
=======

- docs/ Contains all documentation for Korali source and website
- examples/ Contains example scripts that solve all of Korali's problem types
- python/ Contains Korali's pure python source code
- source/ Contains Korali's C++ source code (python extension)
- subprojects/ Contains external dependencies required by Korali
- tests/ Contains test scripts to verify Korali's correctness
- tools/ Contains Korali's additional tools and scripts

Contact us
==========

The Korali Project is developed and maintained by

* `Sergio Miguel Martin <https://www.cse-lab.ethz.ch/member/sergio-martin/>`_, martiser at ethz.ch
* `Daniel Waelchli <https://www.cse-lab.ethz.ch/member/daniel-walchli/>`_, wadaniel at ethz.ch
* `Georgios Arampatzis <https://www.cse-lab.ethz.ch/member/georgios-arampatzis/>`_, garampat at ethz.ch
* `Pascal Weber <https://www.cse-lab.ethz.ch/member/pascal-weber/>`_, webepasc at ethz.ch

Frequent contributors: Fabian Wermelinger, Lucas Amoudrouz, Ivica Kicic

