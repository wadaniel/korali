# Korali json hierarchy

- [Problem](#problem)

- [Parameters](#parameters)

- [Solver](#solver)

- [Seed](#seed)

- [Verbosity](#verbosity)




## Problem

- Objective

    **Type:** String

    **Values:**

    * `Direct Evaluation`

    * `Posterior`


- Reference Data

    **Type:** Double array

    **Values:** Real values





## Parameters

- Name

    **Type:** String

    **Values:** A user specified string.


- Distribution

    **Type:** String

    **Values:**

    * `Uniform`
    * `Gaussian`
    * `Exponential`
    * `Laplace`
    * `Gamma`
    * `Cauchy`

- Type

    **Type:** String

    **Values:**

    * `Computational` Any parameter that is related to the *computational* model

    * `Statistical` Any parameter that is involved in the *statistical* model, e.g. in
    the definition of the likelihood function.




- Initial Mean

- Minimum

- Maximum


## Solver

- Method

    **Type:** String

    **Values:**

    - [`CMA-ES`](#cma-es)

    - [`TMCMC`](#tmcmc)

#### CMA-ES

- Lambda

    **Description:** Population size

    **Type:** Integer

    **Default:**

- Termination Criteria

    - Max Generations

        **Description:**

        **Type:**

        **Default:**

    - Min Fitness

        **Description:**

        **Type:**

        **Default:**

    - Max Model Evaluations

        **Description:**

        **Type:**

        **Default:**

    - Fitness Diff Threshold

        **Description:**

        **Type:**

        **Default:**

    - Min DeltaX

        **Description:**

        **Type:**

        **Default:**

    - Max Standard Deviation

        **Description:**

        **Type:**

        **Default:**

    - Max Condition Covariance

        **Description:**

        **Type:**

        **Default:**

    - Ignore

        **Description:**

        **Type:**

        **Default:**



#### TMCMC

- Population Size

    **Description:**

    **Type:**

    **Default:**

- Covariance Scaling

    **Description:**

    **Type:**

    **Default:**

- Burn In

    **Description:**

    **Type:**

    **Default:**

- Coefficient of Variation

    **Description:**

    **Type:**

    **Default:**

- Min Rho Update

    **Description:**

    **Type:**

    **Default:**

- Use Local Covariance

    **Description:**

    **Type:**

    **Default:**
