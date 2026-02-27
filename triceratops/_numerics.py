"""
Numerically stable helper functions used by marginal_likelihoods.py.

Kept in a separate module so the pure-numerics logic can be imported and
tested without pulling in the full triceratops astronomy dependency chain.
"""

import numpy as np
from scipy.special import logsumexp as _logsumexp


def _log_mean_exp(logw: np.ndarray, *, N_total: int) -> float:
    """Numerically stable log(mean(exp(logw))).

    Equivalent to log(Z) = log(mean(exp(lnL))) but avoids float underflow
    when lnL values are very negative (e.g. long light curves with many
    flux points).

    Non-finite entries (``-inf`` for non-transiting draws, ``NaN`` for
    PyTransit numerical failures) contribute zero weight to the sum but
    still count in the denominator, preserving the same mean semantics as
    ``np.mean(np.nan_to_num(np.exp(lnL + 600)))``.

    Args:
        logw: 1-D array of log-weight values (e.g. lnL or lnL + lnprior).
        N_total: Total number of MC draws (len(logw) before any filtering).
                 Must equal len(logw). Exposed as a keyword argument to make
                 the intent explicit at every call site.
    Returns:
        log(mean(exp(logw))) as a Python float, or -inf if all entries are
        non-finite.
    """
    finite = np.isfinite(logw)
    if not np.any(finite):
        return -np.inf
    return float(_logsumexp(logw[finite]) - np.log(N_total))
