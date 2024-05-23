Background
==========

CATE estimation
---------------

TODO

MetaLearners
------------
Following the definition by `Kunzel et al. <https://doi.org/10.1073/pnas.1804597116>`_, MetaLearning
refers to any technique where a functional learning problem is decomposed into a sequence of
simpler learning tasks.

There maybe several theoretical or practical reasons to prefer one CATE decomposition over
another, which is why so many CATE metalearners have been developed. Most of them are
developed for the binary treatment case although extensions to the multiple discrete
treatments can be implemented. In this library we assume we have :math:`K` treatment
variants and a control variant, i.e. :math:`w \in \{0,\dots,K\}`. Given this scenario,
we estimate the CATE for each variant in relation to the control:

.. math::
    \tau_k(x) := \mathbb{E}[Y(k) - Y(0) | X=x] \; \forall k \in \{1,\dots, K\}

S-Learner
"""""""""""""""""""""
The S-Learner was introduced by `Kuenzel et al. (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.
In this case the treatment indicator is included as a feature similar to all other features
and estimating the combined response function:

.. math::
    \mu (x, w) := \mathbb{E}[Y | X = x, W=w]

using any base learner.

Then the CATE is estimated as:

.. math::
    \hat{\tau}^S(x) := \hat{\mu}(x,1) - \hat{\mu}(x,0)

In the case of multiple discrete treatments the treatment variant is encoded in one
column if the model natively supports categorical variables or one-hot-encoded if it does
not have native support.
The CATE for each treatment variant against the control is then estimated with:

.. math::
    \hat{\tau}_k^S(x) = \hat{\mu}(x,k) - \hat{\mu}(x,0) \; \forall k \in \{1,\dots, K\}

T-Learner
"""""""""""""""""""""
The T-Learner was introduced by `Kuenzel et al. (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.
In the T-Learner the conditional average outcomes are estimated using one estimator for
each treatment variant:

.. math::
    \mu_0 (x) &:= \mathbb{E}[Y(0) | X = x] \\
    \mu_1 (x) &:= \mathbb{E}[Y(1) | X = x]

:math:`\hat{\mu}_0` and :math:`\hat{\mu}_1` are estimated using the untreated and treated observations
respectively.

Then the CATE is estimated as:

.. math::
    \hat{\tau}^T(x) := \hat{\mu}_1(x) - \hat{\mu}_0(x)

In the case of multiple discrete treatments one estimator is trained for each treatment
variant (including the control):

.. math::
    \mu_k (x) := \mathbb{E}[Y(k) | X = x] \; \forall k \in \{0,\dots, K\}

The CATE for each treatment variant against the control is then estimated with:

.. math::
    \hat{\tau}_k^T(x) := \hat{\mu}_k(x) - \hat{\mu}_0(x) \; \forall k \in \{1,\dots, K\}

X-Learner
"""""""""""""""""""""
The X-Learner was introduced by `Kuenzel et al. (2019) <https://arxiv.org/pdf/1706.03461.pdf>`_.
It is an extension of the T-Learner and consists of three stages:

#.  Estimate the conditional average outcomes for each variant:

    .. math::
        \mu_0 (x) &:= \mathbb{E}[Y(0) | X = x] \\
        \mu_1 (x) &:= \mathbb{E}[Y(1) | X = x]

#.  Impute the treatment effect for the observations in the treated group based on the
    control-outcome estimator as well as the treatment effect for the observations in the control
    group based on the treatment-outcome estimator:

    .. math::
        \widetilde{D}_1^i &:= Y^i_1 - \hat{\mu}_0(X^i_1) \\
        \widetilde{D}_0^i &:= \hat{\mu}_1(X^i_0) - Y^i_0

    Then estimate :math:`\tau_1(x) := \mathbb{E}[\widetilde{D}^i_1 | X]` and
    :math:`\tau_0(x) := \mathbb{E}[\widetilde{D}^i_0 | X]` using the observations in the
    treatment group and the ones in the control group respectively.
#.  Define the CATE estimate by a weighted average of the two estimates in stage 2:

    .. math::
        \hat{\tau}^X(x) := g(x)\hat{\tau}_0(x) + (1-g(x))\hat{\tau}_1(x)

    where :math:`g(x) \in [0,1]`. We take :math:`g(x) := \mathbb{E}[W = 1 | X]` to be
    the propensity score.

TODO: multitreatment

R-Learner
"""""""""""""""""""""
The R-Learner was introduced by `Nie et al. (2017) <https://arxiv.org/pdf/1712.04912>`_.
It consists of two stages:

#.  Estimate a general outcome model and a propensity model:

    .. math::
        m(x) &:= \mathbb{E}[Y | X=x] \\
        e(x) &:= \mathbb{E}[W = 1 | X=x]

#.  Estimate the treatment effect by minimising the R-Loss:

    .. math::
        \DeclareMathOperator*{\argmin}{arg\,min}
        \hat{\tau}^R (x) &:= \argmin_{\tau}\Bigg\{\mathbb{E}\Bigg[\bigg(\left\{Y^i - \hat{m}(X^i)\right\} - \left\{W^i - \hat{e}(X^i)\right\}\tau(X^i)\bigg)^2\Bigg]\Bigg\} \\
        &=\argmin_{\tau}\left\{\mathbb{E}\left[\left\{W^i - \hat{e}(X^i)\right\}^2\bigg(\frac{\left\{Y^i - \hat{m}(X^i)\right\}}{\left\{W^i - \hat{e}(X^i)\right\}} - \tau(X^i)\bigg)^2\right]\right\} \\
        &= \argmin_{\tau}\left\{\mathbb{E}\left[{\tilde{W}^i}^2\bigg(\frac{\tilde{Y}^i}{\tilde{W}^i} - \tau(X^i)\bigg)^2\right]\right\}

    And therefore any ML model which supports weighting each observation differently can be used for the final model.

TODO: multitreatment

DR-Learner
"""""""""""""""""""""
The DR-Learner was introduced by `Kennedy (2020) <https://arxiv.org/pdf/2004.14497>`_.
It consists of two stages:

#.  Estimate  the conditional average outcomes for each variant and a propensity model:

    .. math::
        \mu_0 (x, w) &:= \mathbb{E}[Y(0) | X = x] \\
        \mu_1 (x, w) &:= \mathbb{E}[Y(1) | X = x] \\
        e(x) &:= \mathbb{E}[W = 1 | X=x]

    and construct the pseudo-outcomes:

    .. math::
        \varphi(X^i, W^i, Y^i) := \frac{W^i - \hat{e}(X^i)}{\hat{e}(X^i)(1-\hat{e}(X^i))}\big\{Y^i - \hat{\mu}_{W^i}(X^i)\big\} + \hat{\mu}_{1}(X^i) - \hat{\mu}_{0}(X^i)

#.  Estimate the CATE by regressing :math:`\varphi` on :math:`X`:

    .. math::
        \hat{\tau}^{DR}(x) := \mathbb{E}[\varphi(X^i, W^i, Y^i) | X^i]

TODO: multitreatment
