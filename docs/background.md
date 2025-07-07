# Background

## CATE estimation

This library is all about MetaLearners, and MetaLearners are a popular choice to estimate CATEs. Therefore, we will discuss what CATEs are as well as when and how they can be useful.

### What are CATEs?

CATEs, short for Conditional Average Treatment Effects, are a concept from the field of Causal Inference. They rely on the [potential outcomes framework](https://web.archive.org/web/20150513202106/http://sekhon.berkeley.edu/papers/SekhonOxfordHandbook.pdf) proposed by Rubin and Neyman.

#### Potential outcomes

If, for instance, two treatment variants can be chosen—i.e. $W \in \{0, 1\}$—said potential outcomes framework defines the potential outcomes $Y(1)$ and $Y(0)$, both indicating the outcome if the respective treatment variant had been chosen. The notion of these potential outcomes generalizes to a scenario with $K > 2$ treatment variants where $W \in \{0, \dots, K-1\}$.

Given these potential outcomes per treatment variant, we can investigate the pairwise differences of the potential outcomes to define the notion of treatment effects.

#### Treatment effects

If we think of $Y(1)$ as the outcome if we had chosen treatment variant 1 and $Y(0)$ as the outcome if we had chosen treatment variant 0, it follows naturally that the difference of both $Y(1) - Y(0)$ quantifies the impact on the outcome of the treatment 1 compared to treatment 0. This quantity is often referred to as a treatment effect.

More concretely, these treatment effects can be defined for several levels of granularity. Assuming treatment variants 1 and 0, we distinguish the following:

- The **Individual Treatment Effect** is the treatment effect of one treatment variant compared to another variant on a particular experiment unit $i$, e.g. $\tau^i = Y^i(1) - Y^i(0)$.

- The **Conditional Average Treatment Effect** is an expected treatment effect conditioning on covariates $X$: $\tau(X) = \mathbb{E}[Y(1) - Y(0) | X]$. Since the CATE conditions on covariates, it is able to capture heterogeneity of the treatment effect. In other words, it captures for which instantiations of $X$ the treatment effect is higher and for which it is lower.

- The **Average Treatment Effect** is an expected treatment effect, not conditioning on covariates: $\tau(X) = \mathbb{E}[Y(1) - Y(0)]$.

### CATE estimation is not supervised learning

MetaLearners rely on off-the-shelf supervised Machine Learning models to estimate CATEs. Yet, regular supervised learning problems, i.e., regression and classification problems, come with labels. Thanks to these labels, said Machine Learning models can learn a function fitting features or covariates $X$ to outcomes $Y$. In an ideal world, we could do the same for CATE estimation by replacing $Y$ with the individual treatment effect $\tau^i$ from above.

#### The fundamental problem of Causal Inference

Let's illustrate why doing just that isn't quite as easy with an example.

Imagine we have 3 experiment participants, i.e., experiment units: Susan, Judea, and Victor. Now we would like to figure out the treatment effect of making them listen to Bach, $W = 1$, compared to not making them listen to Bach, $W=0$, on their joyfulness $Y$. There are two covariates that we can base our heterogeneity on:

- Their age
- Whether they are a musician themselves or not

In an ideal world, we'd observe both $Y^i(1)$ and $Y^i(0)$ and could therefore compute $\tau^i$ for each unit, see the table below.

| unit $i$ | age $x_1^i$ | is_musician $x_2^i$ | $Y^i(0)$ | $Y^i(1)$ | $\tau^i$ |
| -------- | ----------- | ------------------- | -------- | -------- | -------- |
| Susan    | 28          | 1                   | 0.6      | 0.8      | 0.2      |
| Judea    | 37          | 1                   | 0.5      | 0.6      | 0.1      |
| Victor   | 42          | 0                   | 0.9      | 0.6      | -0.3     |

If we did indeed have access to this data, we could train a regressor mapping $X$ to $\tau$, thereby estimating the CATE.

Yet, in reality, we never have access to several potential outcomes at once. We can, for instance, make Susan listen to Bach for a while and then not listen to Bach and measure her joyfulness during each period. Yet, these periods are not perfectly equivalent; maybe she's more tired during one of the periods; maybe she still draws from the joys of listening to Bach when not listening to Bach anymore. This aspect of only ever having access to one of the potential outcomes is referred to as the fundamental problem of Causal Inference.

As a consequence of this problem, data from a real-world experiment would rather look as such:

| unit $i$ | age $x_1^i$ | is_musician $x_2^i$ | treatment $W^i$ | $Y^i(0)$ | $Y^i(1)$ | $\tau^i$ |
| -------- | ----------- | ------------------- | --------------- | -------- | -------- | -------- |
| Susan    | 28          | 1                   | 1               | ?        | 0.8      | ?        |
| Judea    | 37          | 1                   | 0               | 0.5      | ?        | ?        |
| Victor   | 42          | 0                   | 1               | ?        | 0.6      | ?        |

Note that there is one treatment assignment $W^i$ for each unit and only the outcome linked to this treatment variant can be observed.

Hence, CATE estimation is not a regular supervised learning problem.

## How can CATEs be useful?

CATEs can be very useful to quantify how well past treatment assignment mechanisms have been doing in light of a given outcome. Yet, they can also be used to prescribe a way of intervening in a forward-looking manner. This 'way of intervening' we call a policy.

More precisely, we define a policy as a mapping from the covariate space $\mathcal{X}$ to the set of treatment variants. The set of treatment variants is also referred to as the action space, set of arms, or option space in other frameworks. Note that a policy could also be probabilistic, i.e., a distribution over treatment variants given covariates. We care about deterministic policies only.

### Learning a policy

If one is given CATE estimates $\tau_{k}(X^i)$, i.e., quantifications of the effect of certain treatment variants on the outcome, compared to other treatment variants for fixed covariates, an optimal policy $\pi$ can be trivially defined:

$$
\pi(X^i) := \arg\max_k \tau_{k}(X^i)
$$

As a consequence, when encountering 'new' data, which hasn't been used for learning a CATE model, we can apply our CATE model on it and assess what treatment variant it should receive.

## When can CATEs be useful?

As was described before, CATEs lend themselves fairly naturally to use cases where:

- There is a notion of a treatment, intervention, or action
- One suspects the treatments to behave heterogeneously with respect to some covariates
- The heterogeneity crosses a decision boundary

In the following image, we see some CATE estimates for an intervention based on a single covariate: age.

![alternate text](imgs/heterogeneity.svg)

We see that in the left image, there is fairly little heterogeneity in the CATE estimates with respect to age. The second image conveys the presence of a lot of heterogeneity of the treatment with respect to age. Yet, this heterogeneity is not relevant in light of a policy definition since all estimates are on 'one side' of the decision boundary, here chosen to be 0. The third picture, on the other hand, illustrates a scenario where heterogeneity can be leveraged for policy learning: in some regions, the CATE is negative and therefore treatment variant 0 should be preferred over treatment variant 1—in other regions, the opposite holds true.

We would like to learn such policies to apply them to previously unseen data. In order to learn the policy, we can use data from an experiment. We can distinguish two cases when it comes to experiment data: [observational](glossary.md#randomized-control-trial-rct) or [RCT](glossary.md#randomized-control-trial-rct) data.

Importantly, MetaLearners for CATE estimation can, in principle, be used for both observational or RCT data. Yet, the following conditions need to be validated in order for the MetaLearners to produce valid estimates:

- **Positivity/overlap**

$$
\forall k: \Pr[W=k|X] > 0
$$

- **Conditional ignorability/unconfoundedness**

$$
\forall k', k''\ s.t.\ k' \neq k: (Y(k'), Y(k'')) \perp W | X
$$

- **Stable Unit Treatment Value**

$$
\forall k: W = k \Rightarrow Y = Y(k)
$$

Where $k$ represents a treatment variant. If the experiment data stems from an RCT, the first two conditions are already met. For more details see [Athey and Imbens (2016)](https://arxiv.org/pdf/1607.00698).

## MetaLearners

Following the definition by [Kunzel et al.](https://doi.org/10.1073/pnas.1804597116), MetaLearning refers to any technique where a functional learning problem is decomposed into a sequence of simpler learning tasks.

There may be several theoretical or practical reasons to prefer one CATE decomposition over another, which is why so many CATE MetaLearners have been developed. Most of them are developed for the binary treatment case, although extensions to multiple discrete treatments can be implemented. In this library, we assume we have $K$ treatment variants including the control variant, i.e., $w \in \{0,\dots,K-1\}$. Given this scenario, we estimate the CATE for each variant in relation to the control:

$$
\tau_k(x) := \mathbb{E}[Y(k) - Y(0) | X=x] \; \forall k \in \{1,\dots, K-1\}
$$

### S-Learner

The S-Learner was introduced by [Kuenzel et al. (2019)](https://arxiv.org/pdf/1706.03461.pdf). In this case, the treatment indicator is included as a feature similar to all other features and estimating the combined response function:

$$
\mu (x, w) := \mathbb{E}[Y | X = x, W=w]
$$

using any base learner.

Then the CATE is estimated as:

$$
\hat{\tau}^S(x) := \hat{\mu}(x,1) - \hat{\mu}(x,0)
$$

#### More than binary treatment

In the case of multiple discrete treatments, the treatment variant is encoded in one column if the model natively supports categorical variables or one-hot-encoded if it does not have native support.

The CATE for each treatment variant against the control is then estimated with:

$$
\hat{\tau}_k^S(x) = \hat{\mu}(x,k) - \hat{\mu}(x,0) \; \forall k \in \{1,\dots, K-1\}
$$

### T-Learner

The T-Learner was introduced by [Kuenzel et al. (2019)](https://arxiv.org/pdf/1706.03461.pdf). In the T-Learner, the conditional average outcomes are estimated using one estimator for each treatment variant:

$$
\begin{align*} % Use the align environment for several lines separated by double backslash
\mu_0 (x) &:= \mathbb{E}[Y(0) | X = x] \\
\mu_1 (x) &:= \mathbb{E}[Y(1) | X = x]
\end{align*}
$$

$\hat{\mu}_0$ and $\hat{\mu}_1$ are estimated using the untreated and treated observations respectively.

Then the CATE is estimated as:

$$
\hat{\tau}^T(x) := \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

#### More than binary treatment

In the case of multiple discrete treatments, one estimator is trained for each treatment variant (including the control):

$$
\mu_k (x) := \mathbb{E}[Y(k) | X = x] \; \forall k \in \{0,\dots, K-1\}
$$

The CATE for each treatment variant against the control is then estimated with:

$$
\hat{\tau}_k^T(x) := \hat{\mu}_k(x) - \hat{\mu}_0(x) \; \forall k \in \{1,\dots, K-1\}
$$

### X-Learner

The X-Learner was introduced by [Kuenzel et al. (2019)](https://arxiv.org/pdf/1706.03461.pdf). It is an extension of the T-Learner and consists of three stages:

1. Estimate the conditional average outcomes for each variant:

    $$
    \begin{align*}
    \mu_0 (x) &:= \mathbb{E}[Y(0) | X = x] \\
    \mu_1 (x) &:= \mathbb{E}[Y(1) | X = x]
    \end{align*}
    $$

1. Impute the treatment effect for the observations in the treated group based on the control-outcome estimator as well as the treatment effect for the observations in the control group based on the treatment-outcome estimator:

    $$
    \begin{align*}
    \widetilde{D}_1^i &:= Y^i_1 - \hat{\mu}_0(X^i_1) \\
    \widetilde{D}_0^i &:= \hat{\mu}_1(X^i_0) - Y^i_0
    \end{align*}
    $$

    Then estimate \(\tau_1(x) := \mathbb{E}[\widetilde{D}_1^i | X=x]\) and \(\tau_0(x) := \mathbb{E}[\widetilde{D}_0^i | X=x]\) using the observations in the treatment group and the ones in the control group respectively.

1. Define the CATE estimate by a weighted average of the two estimates in stage 2:

    $$
    \hat{\tau}^X(x) := g(x)\hat{\tau}_0(x) + (1-g(x))\hat{\tau}_1(x)
    $$

    Where \(g(x) \in [0,1]\). We take \(g(x) := \mathbb{E}[W = 1 | X=x]\) to be the propensity score.

#### More than binary treatment

In the case of multiple discrete treatments, the stages are similar to the binary case:

1. One outcome model is estimated for each variant (including the control), and one propensity model is trained as a multiclass classifier, $\forall k \in \{0,\dots, K-1\}$:

    $$
    \begin{align*}
    \mu_k (x) &:= \mathbb{E}[Y(k) | X = x]\\
    e(x, k) &:= \mathbb{E}[\mathbb{I}\{W = k\} | X=x] = \mathbb{P}[W = k | X=x]
    \end{align*}
    $$

1. The treatment effects are imputed using the corresponding outcome estimator, $\forall k \in \{1,\dots, K-1\}$:

    $$
    \begin{align*}
    \widetilde{D}_k^i &:= Y^i_k - \hat{\mu}_0(X^i_k) \\
    \widetilde{D}_{0,k}^i &:= \hat{\mu}_k(X^i_ 0) - Y^i_0
    \end{align*}
    $$

    Then $\tau_k(x) := \mathbb{E}[\widetilde{D}^i_k | X=x]$ is estimated using the observations which received treatment $k$ and $\tau_{0,k}(x) := \mathbb{E}[\widetilde{D}^i_{0,k} | X=x]$ using the observations in the control group.

1. Finally, the CATE for each variant is estimated as a weighted average:

    $$
    \hat{\tau}_k^X(x) := g(x, k)\hat{\tau}_{0,k}(x) + (1-g(x,k))\hat{\tau}_k(x)
    $$

    Where

    $$
    g(x,k) := \frac{\hat{e}(x,k)}{\hat{e}(x,k) + \hat{e}(x,0)}
    $$

### R-Learner

The R-Learner was introduced by [Nie et al. (2017)](https://arxiv.org/pdf/1712.04912). It consists of two stages:

1. Estimate a general outcome model and a propensity model:

    $$
    \begin{align*}
    m(x) &:= \mathbb{E}[Y | X=x] \\
    e(x) &:= \mathbb{P}[W = 1 | X=x]
    \end{align*}
    $$

1. Estimate the treatment effect by minimizing the R-Loss:

    $$
    \begin{align*}
    \hat{\tau}^R (\cdot) &:= \argmin_{\tau}\Bigg\{\mathbb{E}\Bigg[\bigg(\left\{Y^i - \hat{m}(X^i)\right\} - \left\{W^i - \hat{e}(X^i)\right\}\tau(X^i)\bigg)^2\Bigg]\Bigg\} \\
    &=\argmin_{\tau}\left\{\mathbb{E}\left[\left\{W^i - \hat{e}(X^i)\right\}^2\bigg(\frac{\left\{Y^i - \hat{m}(X^i)\right\}}{\left\{W^i - \hat{e}(X^i)\right\}} - \tau(X^i)\bigg)^2\right]\right\} \\
    &= \argmin_{\tau}\left\{\mathbb{E}\left[{\widetilde{W}^i}^2\bigg(\frac{\widetilde{Y}^i}{\widetilde{W}^i} - \tau(X^i)\bigg)^2\right]\right\}
    \end{align*}
    $$

    Where

    $$
    \begin{align*}
    \widetilde{W}^i &= W^i - \hat{e}(X^i) \\
    \widetilde{Y}^i &= Y^i - \hat{m}(X^i)
    \end{align*}
    $$

    And therefore any ML model which supports weighting each observation differently can be used for the final model.

#### More than binary treatment

In the case of multiple discrete treatments, the stages are similar to the binary case. More precisely, the first stage is perfectly equivalent. Yet, the second stage includes a conceptual change: we arbitrarily define one treatment variant as control—the variant with index 0—and estimate pairwise treatment effects of every other variant to the control variant.

1. Estimate a general outcome model and a propensity model:

    $$
    \begin{align*}
    m(x) &:= \mathbb{E}[Y | X=x] \\
    e(x) &:= \mathbb{P}[W = k | X=x]
    \end{align*}
    $$

1. For each $k \neq 0$, estimate the pairwise treatment effect $\hat{\tau}_{0,k}^R$ between 0 and $k$ by minimizing the R-Loss from above. In order to fit these models, we fit the pseudo outcomes only on observations of either the control group or the treatment variant group $k$.

Note that:

- In chapter 7, [Nie et al. (2017)](https://arxiv.org/pdf/1712.04912) suggest a generalization of the R-Loss simultaneously taking all treatment variants into account. Yet, [Acharki et al. (2023)](https://arxiv.org/pdf/2205.14714) point out practical shortcomings of this approach.

- Our implementation differs subtly from the CausalML implementation: while we train a multi-class propensity model whose estimates we normalize subsequently, CausalML estimates one propensity model per control-treatment pair.

- Rather than estimating one treatment effect per control-treatment pair, we could also estimate the treatment effects between each treatment variant.

### DR-Learner

The DR-Learner was introduced by [Kennedy (2020)](https://arxiv.org/pdf/2004.14497). It consists of two stages:

1. Estimate the conditional average outcomes for each variant and a propensity model:

    $$
    \begin{align*}
    \mu_0 (x, w) &:= \mathbb{E}[Y(0) | X = x] \\
    \mu_1 (x, w) &:= \mathbb{E}[Y(1) | X = x] \\
    e(x) &:= \mathbb{E}[W = 1 | X=x]
    \end{align*}
    $$

    And construct the pseudo-outcomes:

    $$
    \begin{align*}
    \varphi(X^i, W^i, Y^i) := \frac{W^i - \hat{e}(X^i)}{\hat{e}(X^i)(1-\hat{e}(X^i))} \big\{Y^i - \hat{\mu}_{W^i}(X^i)\big\} + \hat{\mu}_{1}(X^i) - \hat{\mu}_{0}(X^i)
    \end{align*}
    $$

1. Estimate the CATE by regressing $\varphi$ on $X$:

    $$
    \hat{\tau}^{DR}(x) := \mathbb{E}[\varphi(X^i, W^i, Y^i) | X^i=x]
    $$

#### More than binary treatment

In the case of multiple discrete treatments, the stages are similar to the binary case:

1. One outcome model is estimated for each variant (including the control), and one propensity model is trained as a multiclass classifier, $\forall k \in \{0,\dots, K-1\}$:

    $$
    \begin{align*}
    \mu_k (x) &:= \mathbb{E}[Y(k) | X = x]\\
    e(x, k) &:= \mathbb{E}[\mathbb{I}\{W = k\} | X=x] = \mathbb{P}[W = k | X=x]
    \end{align*}
    $$

    The pseudo-outcomes are constructed for each treatment variant, $\forall k \in \{1,\dots, K-1\}$:

    $$
    \begin{align*}
    \varphi_k(X^i, W^i, Y^i) := &\frac{Y^i - \hat{\mu}_{k}(X^i)}{\hat{e}(k, X^i)}\mathbb{I}\{W^i = k\} + \hat{\mu}_k(X^i) \\
    &- \frac{Y^i - \hat{\mu}_{0}(X^i)}{\hat{e}(0, X^i)}\mathbb{I}\{W^i = 0\} - \hat{\mu}_0(X^i)
    \end{align*}
    $$
    1. Finally, the CATE is estimated by regressing $\varphi_k$ on $X$ for each treatment variant, $\forall k \in \{1,\dots, K-1\}$:

    $$
    \hat{\tau}_k^{DR}(x) := \mathbb{E}[\varphi_k(X^i, W^i, Y^i) | X^i=x]
    $$
