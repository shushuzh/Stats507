import numpy as np
import pandas as pd
from math import floor
from scipy.stats import norm, binom, beta
from warnings import warn

def ci_mean(
    x,
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]"
):
    """
    Construct an estimate and confidence interval for the mean of `x`.

    Parameters
    ----------
    x : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector from which to form the estimates.
    level : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    str_fmt: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]".

    Returns
    -------
    By default, the function returns a string with a 95% confidence interval
    in the form "mean [95% CI: (lwr, upr)]". A dictionary containing the mean,
    confidence level, lower, bound, and upper bound can also be returned.

    """
    # check input
    try:
        x = np.asarray(x)  # or np.array() as instructed.
    except TypeError:
        print("Could not convert x to type ndarray.")

    # construct estimates
    xbar = np.mean(x)
    se = np.std(x, ddof=1) / np.sqrt(x.size)
    z = norm.ppf(1 - (1 - level) / 2)
    lwr, upr = xbar - z * se, xbar + z * se
    out = {"mean": xbar, "level": 100 * level, "lwr": lwr, "upr": upr}
    # format output
    if str_fmt is None:
        return(out)
    else:
        return(str_fmt.format_map(out))


def ci_prop(
    x,
    level=0.95,
    str_fmt="{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]",
    method="Normal"
):
    """
    Construct point and interval estimates for a population proportion.

    The "method" argument controls the estimates returned. Available methods
    are "Normal", to use the normal approximation to the Binomial, "CP" to
    use the Clopper-Pearson method, "Jeffrey" to use Jeffery's method, and
    "AC" for the Agresti-Coull method.

    By default, the function returns a string with a 95% confidence interval
    in the form "mean [level% CI: (lwr, upr)]". Set `str_fmt=None` to return
    a dictionary containing the mean, confidence level (%-scale, level),
    lower bound (lwr), and upper bound (upr) can also be returned.

    Parameters
    ----------
    x : A 1-dimensional NumPy array or compatible sequence type (list, tuple).
        A data vector of 0/1 or False/True from which to form the estimates.
    level : float, optional.
        The desired confidence level, converted to a percent in the output.
        The default is 0.95.
    str_fmt: str or None, optional.
        If `None` a dictionary with entries `mean`, `level`, `lwr`, and
        `upr` whose values give the point estimate, confidence level (as a %),
        lower and upper confidence bounds, respectively. If a string, it's the
        result of calling the `.format_map()` method using this dictionary.
        The default is "{mean:.1f} [{level:0.f}%: ({lwr:.1f}, {upr:.1f})]".
    method: str, optional
        The type of confidence interval and point estimate desired.  Allowed
        values are "Normal" for the normal approximation to the Binomial,
        "CP" for a Clopper-Pearson interval, "Jeffrey" for Jeffrey's method,
        or "AC" for the Agresti-Coull estimates.

    Returns
    -------
    A string with a (100 * level)% confidence interval in the form
    "mean [(100 * level)% CI: (lwr, upr)]" or a dictionary containing the
    keywords shown in the string.

    """
    # check input type
    try:
        x = np.asarray(x)  # or np.array() as instructed.
    except TypeError:
        print("Could not convert x to type ndarray.")

    # check that x is bool or 0/1
    if x.dtype is np.dtype('bool'):
        pass
    elif not np.logical_or(x == 0, x == 1).all():
        raise TypeError("x should be dtype('bool') or all 0's and 1's.")

    # check method
    assert method in ["Normal", "CP", "Jeffrey", "AC"]

    # determine the length
    n = x.size

    # compute estimate
    if method == 'AC':
        z = norm.ppf(1 - (1 - level) / 2)
        n = (n + z ** 2)
        est = (np.sum(x) + z ** 2 / 2) / n
    else:
        est = np.mean(x)

    # warn for small sample size with "Normal" method
    if method == 'Normal' and (n * min(est, 1 - est)) < 12:
        warn(Warning(
            "Normal approximation may be incorrect for n * min(p, 1-p) < 12."
        ))

    # compute bounds for Normal and AC methods
    if method in ['Normal', 'AC']:
        se = np.sqrt(est * (1 - est) / n)
        z = norm.ppf(1 - (1 - level) / 2)
        lwr, upr = est - z * se, est + z * se

    # compute bounds for CP method
    if method == 'CP':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s, n - s + 1)
        upr = beta.ppf(1 - alpha / 2, s + 1, n - s)

    # compute bounds for Jeffrey method
    if method == 'Jeffrey':
        alpha = 1 - level
        s = np.sum(x)
        lwr = beta.ppf(alpha / 2, s + 0.5, n - s + 0.5)
        upr = beta.ppf(1 - alpha / 2, s + 0.5, n - s + 0.5)

    # prepare return values
    out = {"mean": est, "level": 100 * level, "lwr": lwr, "upr": upr}
    if str_fmt is None:
        return(out)
    else:
        return(str_fmt.format_map(out))        
