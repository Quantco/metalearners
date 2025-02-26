# Glossary

#### Base model

A prediction model used within a MetaLearner. See [Kuenzel et al. (2019)](https://arxiv.org/pdf/1706.03461.pdf).

#### Conditional Average Treatment Effect (CATE)

$\tau(X) = \mathbb{E}[Y(1) - Y(0)|X]$ in the binary case and $\tau_{i,j}(X) = \mathbb{E}[Y(i) - Y(j)|X]$ if more than two variants exist. See [Athey et al. (2016)](https://arxiv.org/abs/1607.00698), Chapter 10.

#### Conditional Average Outcomes

$\mathbb{E}[Y_i(w) | X]$ for each treatment variant $w$.

#### Covariates

The features $X$ based on which a CATE is estimated.

#### Double Machine Learning

Similar to the R-Learner, the Double Machine Learning blueprint relies on estimating two nuisance models in its first stage: a propensity model as well as an outcome model. Unlike the R-Learner, the last-stage or treatment effect model might need to be a specific type of estimator. See [Chernozhukov et al. (2016)](https://arxiv.org/abs/1608.00060).

#### Heterogeneous Treatment Effect (HTE)

Synonym for CATE.

#### MetaLearner

CATE model which relies on arbitrary prediction estimators (regressors or classifiers) for the actual estimation. See [Kuenzel et al. (2019)](https://arxiv.org/pdf/1706.03461.pdf).

#### Nuisance model

A first-stage model in a MetaLearner. See [Nie et al. (2019)](https://arxiv.org/pdf/1712.04912.pdf).

#### Observational data

Experiment data collected outside of a RCT, i.e., treatment assignments can depend on covariates or potential outcomes. See [Athey et al. (2016)](https://arxiv.org/abs/1607.00698).

#### Outcome model

A model estimating the outcome based on covariates, i.e., $\mathbb{E}[Y|X]$.

#### Potential outcomes

Outcomes under various variants, e.g., $Y(0)$ and $Y(1)$, in Rubin-Causal Model (RCM). See [Holland et al. (1986)](https://www.cs.columbia.edu/~blei/fogm/2023F/readings/Holland1986.pdf).

#### Propensity model

A model estimating the propensity score.

#### Propensity score

The probability of receiving a certain treatment/variant, conditioning on covariates: $\Pr[W_i = w | X]$. See [Rosenbaum et al. (1983)](https://academic.oup.com/biomet/article/70/1/41/240879?login=false).

#### Randomized Control Trial (RCT)

An experiment in which the treatment assignment is independent of the covariates $X$. See [Athey et al. (2016)](https://arxiv.org/abs/1607.00698).

#### Treatment effect model

A second-stage model in a MetaLearner which models the treatment effects as a function of covariates.
