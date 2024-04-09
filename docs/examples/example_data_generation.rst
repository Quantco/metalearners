Example: Generating data
========================

Motivation
----------

Given the fundamental problem of Causal Inference, simulating or
generating data is of particular relevance when working with ATE or
CATE estimation: it allows to have a ground truth that we don't get
from the real world.

For instance, when generating data, we can have access to the
Individual Treatment Effect and use that ground truth to evaluate a
treatment effect method at hand.

In the following example we will describe how the modules
:mod:`metalearners.data_generation` and
:mod:`metalearners.outcome_functions` can be used to generate data in
light of treatment effect estimation.


How-to
------

In the context of treatment effect estimation, our data usually
consists of 3 ingredients:

- Covariates
- Treatment assignments
- Observed outcomes

In this particular scenario of simulating data, we can add some
quantities of interest which are not available in the real world:

- Potential outcomes
- True CATE or true ITE

Let's generate those quantities one after another.


Covariates
""""""""""

Let's start by generating covariates. We will use
:func:`metalearners.data_generation.generate_covariates` for that
purpose. Running

.. code-block:: python

    from metalearners.data_generation import generate_covariates

    features, categorical_features_idx, n_categories = generate_covariates(
         n_obs=1000,
         n_features=8,
         n_categoricals=3,
         format="pandas",
    )
    features.head()

yields

.. code-block::

              0         1         2         3         4  5  6  7
    0  0.892596  0.085405 -4.344183  2.306697 -5.562215  5  2  1
    1  2.698906 -0.383328 -3.456225  0.455230 -3.938590  3  0  1
    2  0.864039 -1.836730 -1.604536  2.676829 -0.383328  0  0  1
    3 -1.770170 -1.073292 -1.098083  1.604426 -3.966995  4  2  1
    4  1.208203  0.057309 -3.272325  3.071431 -3.087522  0  4  0

We see that we generated a DataFrame with 8 columns of which the last
three are categoricals.


Treatment assignments
"""""""""""""""""""""

In this example we will replicate the setup of an RCT, i.e. where the
treatment assignments are independent of the covariates. We rely on
:func:`metalearners.data_generation.generate_treatment`. Running

.. code-block:: python

    import numpy as np
    from metalearners.data_generation import generate_treatment

    # We use a fair conflip as a reference.
    propensity_scores = .5 * np.ones(1000)
    treatment = generate_treatment(propensity_scores)
    type(treatment), np.unique(treatment), treatment.mean()

yields

.. code-block::

   (<class 'numpy.ndarray'>, array([0, 1]), 0.505)

As we would expect, an array of binary assignments is generated. The
average approximately corresponds to the universal propensity score of
.5.


Potential outcomes
""""""""""""""""""

In this example we will rely on
:func:`metalearners.outcome_functions.linear_treatment_effect`, which
generates additive treatment effects which are linear in the features.
Note that there are other potential outcome functions available. Running

.. code-block:: python

    from metalearners._utils import get_linear_dimension
    from metalearners.outcome_functions import linear_treatment_effect

    dim = get_linear_dimension(features)
    outcome_function = linear_treatment_effect(dim)
    potential_outcomes = outcome_function(features)
    potential_outcomes

yields

.. code-block::

    array([[-2.20741984e+00, -7.01760379e-01],
           [ 3.67600139e-03,  9.24023117e-01],
           [-1.32600557e+00, -1.40630988e+00],
           ...,
           [-7.01070540e+00, -1.24786720e+01],
           [-2.39323972e+00, -3.90233572e+00],
           [-5.31585746e+00, -1.10773973e+01]])

i.e. one column with the potential outcome :math:`Y(0)` and one column
with the potential outcome :math:`Y(1)`. The individual treatment
effect can be inferred as a subtraction of both.


Observed outcomes
"""""""""""""""""

Lastly, we can combine the treatment assignments and potential
outcomes to generate the observed outcomes. Note that there might be
noise which distinguishes the potential outcome from the observed
outcome. For that purpose we can use
:func:`metalearners.data_generation.compute_experiment_outputs` and run

.. code-block:: python

    from metalearners.data_generation import compute_experiment_outputs

    observed_outcomes, true_cate = compute_experiment_outputs(
        potential_outcomes,
        treatment,
    )
