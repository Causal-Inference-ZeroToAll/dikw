"""
上述代码中定义了五个函数，分别对应了五种不同的生成模拟数据的方法：
simulate_nuisance_and_easy_treatment()：这种方法生成的数据，包含了难以处理的干扰项和容易估计的处理效应。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup A 生成的。
simulate_randomized_trial()：这种方法生成的数据，模拟了一个随机试验。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup B 生成的。
simulate_easy_propensity_difficult_baseline()：这种方法生成的数据，模拟了一个易于估计的倾向得分和一个难以估计的基线。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup C 生成的。
simulate_unrelated_treatment_control()：这种方法生成的数据，模拟了一个不相关的处理和对照组。它是根据 Nie X. 和 Wager S. (2018) 的 "Quasi-Oracle Estimation of Heterogeneous Treatment Effects" 中的 Setup D 生成的。
simulate_hidden_confounder()：这种方法生成的数据，模拟了一个隐藏的混淆变量对处理造成偏倚的情况。它是根据 Louizos et al. (2018) 的 "Causal Effect Inference with Deep Latent-Variable Models" 生成的。

这些函数的输出都是一个包含六个元素的元组，分别为 y（因变量）、X（自变量）、w（处理）、tau（处理效应）、b（期望结果）和 e（处理得分）。
通过选择不同的 mode 参数值，我们可以使用这些函数生成不同的模拟数据，用于评估不同的因果推断方法的性能。
"""


import logging

import numpy as np
import pandas as pd
import scipy
from scipy.special import expit, logit

logger = logging.getLogger("dikw")

def to_dataframe(func):
    def wrapper(*args, **kwargs):
        output_dataframe = kwargs.pop("to_dataframe", False)
        tmp = func(*args, **kwargs)
        if output_dataframe:
            y, X, w, tau, b, e = tmp
            df = pd.DataFrame(X)
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            df.columns = feature_names
            df['outcome'] = y
            df['treatment'] = w
            df['treatment_effect'] = tau
            return df
        else:
            return tmp
    return wrapper

@to_dataframe
def synthetic_data(mode=1, n=1000, p=5, sigma=1.0, adj=0.0):
    """ Synthetic data in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        mode (int, optional): mode of the simulation: \
            1 for difficult nuisance components and an easy treatment effect. \
            2 for a randomized trial. \
            3 for an easy propensity and a difficult baseline. \
            4 for unrelated treatment and control groups. \
            5 for a hidden confounder biasing treatment.
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
                     It does not apply to mode == 2 or 3.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    catalog = {
        1: simulate_nuisance_and_easy_treatment,
        2: simulate_randomized_trial,
        3: simulate_easy_propensity_difficult_baseline,
        4: simulate_unrelated_treatment_control,
        5: simulate_hidden_confounder,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog)
    )
    return catalog[mode](n, p, sigma, adj)


def simulate_nuisance_and_easy_treatment(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with a difficult nuisance components and an easy treatment effect
        From Setup A in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    eta = 0.1
    e = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = expit(logit(e) - adj)
    tau = (X[:, 0] + X[:, 1]) / 2

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_randomized_trial(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data of a randomized trial
        From Setup B in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=5)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1], X[:, 2]) + np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )
    e = np.repeat(0.5, n)
    tau = X[:, 0] + np.log1p(np.exp(X[:, 1]))

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_easy_propensity_difficult_baseline(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with easy propensity and a difficult baseline
        From Setup C in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = 2 * np.log1p(np.exp(X[:, 0] + X[:, 1] + X[:, 2]))
    e = 1 / (1 + np.exp(X[:, 1] + X[:, 2]))
    tau = np.repeat(1.0, n)

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_unrelated_treatment_control(n=1000, p=5, sigma=1.0, adj=0.0):
    """Synthetic data with unrelated treatment and control groups.
        From Setup D in Nie X. and Wager S. (2018) 'Quasi-Oracle Estimation of Heterogeneous Treatment Effects'
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): adjustment term for the distribution of propensity, e. Higher values shift the distribution to 0.
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """

    X = np.random.normal(size=n * p).reshape((n, -1))
    b = (
        np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2])
        + np.maximum(np.repeat(0.0, n), X[:, 3] + X[:, 4])
    ) / 2
    e = 1 / (1 + np.exp(-X[:, 0]) + np.exp(-X[:, 1]))
    e = expit(logit(e) - adj)
    tau = np.maximum(np.repeat(0.0, n), X[:, 0] + X[:, 1] + X[:, 2]) - np.maximum(
        np.repeat(0.0, n), X[:, 3] + X[:, 4]
    )

    w = np.random.binomial(1, e, size=n)
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    return y, X, w, tau, b, e


def simulate_hidden_confounder(n=10000, p=5, sigma=1.0, adj=0.0):
    """Synthetic dataset with a hidden confounder biasing treatment.
        From Louizos et al. (2018) "Causal Effect Inference with Deep Latent-Variable Models"
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=3)
        sigma (float): standard deviation of the error term
        adj (float): no effect. added for consistency
    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - y ((n,)-array): outcome variable.
            - X ((n,p)-ndarray): independent variables.
            - w ((n,)-array): treatment flag with value 0 or 1.
            - tau ((n,)-array): individual treatment effect.
            - b ((n,)-array): expected outcome.
            - e ((n,)-array): propensity of receiving treatment.
    """
    z = np.random.binomial(1, 0.5, size=n).astype(np.double)
    X = np.random.normal(z, 5 * z + 3 * (1 - z), size=(p, n)).T
    e = 0.75 * z + 0.25 * (1 - z)
    w = np.random.binomial(1, e)
    b = expit(3 * (z + 2 * (2 * w - 2)))
    y = np.random.binomial(1, b)

    # Compute true ite tau for evaluation (via Monte Carlo approximation).
    t0_t1 = np.array([[0.0], [1.0]])
    y_t0, y_t1 = expit(3 * (z + 2 * (2 * t0_t1 - 2)))
    tau = y_t1 - y_t0
    return y, X, w, tau, b, e



def simulate_base_iv_data(n=1000, p=5, binary_treatment=True, to_daftaframe=True):
    """Synthetic iv data with continuous treatment.
    References: https://github.com/1587causalai/EconML/blob/75b40b6b07ee8aa49e0be75057b54e2458158284/notebooks/OrthoIV%20and%20DRIV%20Examples.ipynb

    """
    X = np.random.normal(0, 1, size=(n, p))
    Z = np.random.binomial(1, 0.5, size=(n,))
    nu = np.random.uniform(0, 5, size=(n,))
    coef_Z = 0.8
    C = np.random.binomial(
        1, coef_Z * scipy.special.expit(0.4 * X[:, 0] + nu)
    )  # Compliers when recomended
    C0 = np.random.binomial(
        1, 0.006 * np.ones(X.shape[0])
    )  # Non-compliers when not recommended
    tmp_T = C * Z + C0 * (1 - Z)
    if not binary_treatment:
        cost = lambda X: 10 * X[:, 1] ** 2
        w = cost(X) * tmp_T
    else:
        w = tmp_T

    true_fn = lambda X: X[:, 0] + 0.5 * X[:, 1] + 0.5 * X[:, 2]
    tau = true_fn(X)

    y = (
            true_fn(X) * w  # 这里意味着 outcome 关于 treatment 是线性的
            + 2 * nu
            + 5 * (X[:, 3] > 0)
            + 0.1 * np.random.uniform(0, 1, size=(n,))
    )

    if to_daftaframe:
        df = pd.DataFrame()
        df['outcome'] = y
        df['treatment'] = w
        df['treatment_effect'] = tau
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        df[feature_names] = X
        df['iv'] = Z
        return df
    else:
        return y, X, w, tau, Z


if __name__== "__main__":
    # simulate_base_iv_data()
    df = simulate_base_iv_data(p=4, to_daftaframe=True)
    print(df.head())