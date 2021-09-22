Reinforcement Learning examples on pettingZoo
==============================================

This folders contain a ready-to-use setup to run pettingZoo. 

Pre-Requisites:
------------------
None.

Running an environment:
-------------------------

Any of the following environments are available for testing:

.. code-block:: bash
   
   % pettingZoo
   Waterworld


To run any of these, use the following example:

.. code-block:: bash

   python3 run-vracer.py --env Waterworld --dis 'Clipped Normal'

Producing a movie:
-------------------------

To generate a movie that displays the outcome of a particular trained policy, use the following command:

.. code-block:: bash

   python3 ./genMovie --env AntBulletEnv-v0 --input _result_vracer_AntBulletEnv-v0 --output myMovie
   
The command will read the result of training an `AntBulletEnv-v0` environment from the `_result_vracer_AntBulletEnv` folder and output an `.mp4` movie in the `myMovie` folder.

