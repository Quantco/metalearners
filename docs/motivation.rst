Motivation
==========

Why CATE estimation?
--------------------

Please see the section :ref:`how-can-cates-be-useful`.

.. _motivation-why-metalearners:

Why MetaLearners?
-----------------

There are various ways for estimating CATEs, such as
`Targeted Maximum Likelihood Estimation <https://academic.oup.com/aje/article/185/1/65/2662306?login=false>`_,
`Causal Forests <https://arxiv.org/abs/1902.07409>`_ or MetaLearners.

We've found MetaLearners to be a particularly compelling approach
for CATE estimation because

- they are conceptually simple
- some of them come with strong statistical guarantees, see e.g.
  `Nie et al. (2019) <https://arxiv.org/pdf/1712.04912.pdf>`_ for the
  R-Learner or `Kennedy (2023) <https://arxiv.org/abs/2004.14497>`_ for the DR-Learner
- they rely on existing, arbitrary prediction approaches

The latter point is particularly important since it implies
that battle-tested and production-grade code from existing prediction
libraries such as ``scikit-learn``, ``lightgbm`` or ``xgboost`` can be
reused. Given that the field of CATE estimation is still young and
engineering efforts limited, this is a highly relevant factor.


.. _motivation-why-not-causalml-or-econml:

Why not ``causalml`` or ``econml``?
-----------------------------------

`causalml <https://github.com/uber/causalml>`_ and `econml
<https://github.com/py-why/EconML>`_ are open-source Python libraries
providing, among other things, implementations of many MetaLearners.

What we've come to like about the design of both is that

* their Metalearner implementations mostly follow the interface one might expect from an ``sklearn`` Estimator
* they are, in the intended use cases, fairly straight-forward and intuitive to use

Yet, we've also found that in some regards, the MetaLearner
implementations from ``causalml`` and ``enconml`` don't perfectly lend
themselves to use cases we care about.

Accessing base models
"""""""""""""""""""""

While MetaLearners are, in principle, designed in a very modular
fashion, we've struggled to access individual base models in a
meaningful way.

One reason to access the base models is to evaluate their individual
performance. Due to the fundamental problem of Causal Inference we
are not able to evaluate a MetaLearner based on a simple metric
measuring the mismatch between estimate and ground truth. Yet, we
might want to do this for our base learners which often do have
ground truth labels to compare the estimates to. Yet, this is not
supported by ``econml`` and ``causalml``.

.. image:: imgs/component_eval.drawio.svg
  :width: 400

In the illustration above, we indicate that we'd like to access,
predict with and evaluate a propensity model -- one base model of
the MetaLearner at hand -- in isolation.

See, for instance, `econml issue 619 <https://github.com/py-why/EconML/issues/619>`_.


Reusing trained base models
"""""""""""""""""""""""""""

Given MetaLearners' modular design, it should in principle be simple
to not only train all base estimators of a MetaLearner 'together' but
to reuse already trained base models.

We envision two concrete use cases where this might be relevant in
that it would save considerable resources:

* When tuning hyperparameters of a given MetaLearner architecture
  (e.g. an R-Learner) on a given dataset, one might, for instance,
  want to tune the hyperparameters of an outcome model in light of
  the behaviour of the overall MetaLearner. In such a scenario, it
  is redundant to retrain a propensity model for every single outcome
  model hyperparameter constellation. Instead, one might want to reuse
  and plug in an already trained propensity model.

* When training several MetaLearner architectures on the same dataset,
  some base models might be part of the design of several of these
  MetaLearner architectures. An example of this could be an outcome
  model, used in both the R-Learner and DR-Learner. In such a
  scenario, it seems desirable to us to reuse the conceptually
  equivalent outcome model instead of training it several times.

 .. image:: imgs/component_reuse.drawio.svg
  :width: 400

The illustration above indicates the intention to reuse an already trained
base estimator as part of a MetaLearner.

See `econml issue 646 <https://github.com/py-why/EconML/issues/646>`_
for reference. The `causalml documentation <https://causalml.readthedocs.io/en/latest/causalml.html#causalml.inference.meta.BaseDRLearner>`_
provides no officially supported way of passing in pre-trained
models. Note that the specified models are first `copied
<https://github.com/uber/causalml/blob/750e84e4916e6ec1f364bd30d5504f9b0e437f93/causalml/inference/meta/drlearner.py#L113-L132>`_
and then `fit <https://github.com/uber/causalml/blob/750e84e4916e6ec1f364bd30d5504f9b0e437f93/causalml/inference/meta/drlearner.py#L150-L203>`_
from scratch.

Working with ``pandas`` DataFrames
""""""""""""""""""""""""""""""""""

Many standard estimation libraries, such as ``sklearn`` or
``lightgbm``, accept ``pandas`` ``DataFrame`` as well as ``numpy``
``ndarrays`` as input - sometimes even generic interfaces such as the
`Array API standard
<https://data-apis.org/array-api/latest/purpose_and_scope.html>`_. Importantly,
a user would not only expect those to be accepted, but also to be
treated in a way that corresponds to their semantics.

Since the operational essence of MetaLearners is merely distributing
the right data (e.g. covariates and outcomes indexed on treated
observations) from the right source (e.g. a base estimator or a raw
input) to the right sink (e.g. a base estimator or final output), we
would expect that anything the base model of
choice can support should also be supported by a MetaLearner library.

Since we are concerned about tabular data, support for ``pandas``
``DataFrame``\s is of particular importance. Now, in most cases,
``econml`` and ``causalml`` accept DataFrames; in many do they work
as intended with them. Yet, under the hood, ``econml`` and
``causalml`` transform every data structure to ``numpy`` (see
`this causalml snippet <https://github.com/uber/causalml/blob/750e84e4916e6ec1f364bd30d5504f9b0e437f93/causalml/inference/meta/drlearner.py#L101>`_
and
`this econml snippet <https://github.com/py-why/EconML/blob/ed4fe33b2ba4e047332c0951c0ed5bfe5b139788/econml/_ortho_learner.py#L747>`_
). Concretely, this leads to
errors with non-integer categoricals and silent errors with integer
categoricals when using
``pandas``\'s
`category dtype
<https://pandas.pydata.org/docs/user_guide/categorical.html>`_ and
``lightgbm`` base models even though ``lightgbm`` can handle the
former just fine. See
`this notebook <https://github.com/kklein/pydata_ams/blob/main/notebooks/categorical_mess.ipynb>`_
for an illustration.

An important illustration of the usefulness of categorical data types
is working with discrete, yet more than binary variants.
Here, ``econml``, for instance, internally encodes these variants with
one-hot encoding. This encoding is not easily undone by the user, and
therefore, results can be cumbersome to interpret.


Using different covariate sets for different base learners
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Most base learners in a MetaLearner expect some covariate matrix
``X``. Conceptually, we need to make sure that this ``X`` satisfies
our fundamental assumptions of positivity, unconfoundedness and stable
unit treatment value. Yet, if we know of certain (conditional)
independences, we might not always require this entire covariate
matrix for each base learner. Conversely, offering a base learner more
features than we know are relevant might make the learning process
more fragile to noise and prone to overfitting.

In the following illustration we indicate that we have a column-wise
partitioning of ``X`` into ``X1`` and ``X2``. One base estimator
relies on ``X1`` only, one on ``X2`` only and one on ``X``,
i.e. ``X1`` and ``X2``.

.. image:: imgs/covariate_sets.drawio.svg
  :width: 400

For this reason, we would want to be able to define which covariate
set is used by which base learner. This is currently not supported by
``econml`` or ``causalml``.



.. _Motivation_multiprocessing:

Multiprocessing training of base learners
"""""""""""""""""""""""""""""""""""""""""

Many MetaLearners come with two 'stages' of base models. The models of
the first stage, nuisance models, are trained independently of each
other. The models of the second stage, the treatment models, are
trained independently of each other, too.

Clearly, this is a perfect setup for concurrent training of
various models which are independent of each other -- trading off space
for time. Yet, neither
``causalml`` nor ``econml`` support multiprocessing within a stage.

See, for instance, `causalml issue 616
<https://github.com/uber/causalml/issues/616>`_.
