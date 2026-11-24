"""ECE 3715 Mini-Project 2 library: continuous random variables.

Every function is vectorized: it takes NumPy arrays and returns NumPy arrays,
so the notebook never loops over individual samples.
"""
import numpy as np
from scipy.special import erfc, erfcinv


# ---------------------------------------------------------------- Q function

def q_function(x):
    x = np.asarray(x, dtype=float)
    return 0.5 * erfc(x / np.sqrt(2.0))


def q_inverse(p):
    p = np.asarray(p, dtype=float)
    return np.sqrt(2.0) * erfcinv(2.0 * p)


def std_normal_cdf(x):
    return 1.0 - q_function(x)


def std_normal_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------- analytic CDFs / pdfs

def cdf_uniform(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def pdf_uniform(x):
    x = np.asarray(x, dtype=float)
    return ((x >= 0) & (x <= 1)).astype(float)


def cdf_exponential(x, lam=1.0):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, -np.expm1(-lam * np.maximum(x, 0)))


def pdf_exponential(x, lam=1.0):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, lam * np.exp(-lam * np.maximum(x, 0)))


def cdf_rayleigh(x, sigma=1.0):
    x = np.asarray(x, dtype=float)
    return np.where(x < 0, 0.0, -np.expm1(-np.maximum(x, 0)**2 / (2 * sigma**2)))


def pdf_rayleigh(x, sigma=1.0):
    x = np.asarray(x, dtype=float)
    xp = np.maximum(x, 0)
    return np.where(x < 0, 0.0, xp / sigma**2 * np.exp(-xp**2 / (2 * sigma**2)))


# ---------------------------------------------------------------- inversion sampling

def inv_cdf_exponential(u, lam=1.0):
    u = np.asarray(u, dtype=float)
    return -np.log1p(-u) / lam


def inv_cdf_rayleigh(u, sigma=1.0):
    u = np.asarray(u, dtype=float)
    return sigma * np.sqrt(-2.0 * np.log1p(-u))


def sample_by_inversion(inv_cdf, n, rng, **kw):
    return inv_cdf(rng.random(n), **kw)


# ---------------------------------------------------------------- goodness of fit

def ks_statistic(samples, cdf):
    x = np.sort(np.asarray(samples, dtype=float))
    n = x.size
    F = cdf(x)
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - F), np.max(F - (i - 1) / n)))


def ks_two_sample(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    Fa = np.searchsorted(a, x, side="right") / a.size
    Fb = np.searchsorted(b, x, side="right") / b.size
    return float(np.max(np.abs(Fa - Fb)))


def ks_critical(n, c=1.36):
    return c / np.sqrt(n)


def ks_critical_two_sample(n, m, c=1.36):
    return c * np.sqrt((n + m) / (n * m))


# ---------------------------------------------------------------- Monte Carlo helpers

def mc_probability(event_count, n_trials):
    p = event_count / n_trials
    return p, np.sqrt(p * (1 - p) / n_trials)


def count_exceedances(draw, threshold, n, rng, chunk=1_000_000):
    hits = 0
    for k in range(0, n, chunk):
        m = min(chunk, n - k)
        hits += int(np.count_nonzero(draw(m, rng) > threshold))
    return hits


def running_mean_var(x):
    x = np.asarray(x, dtype=float)
    n = np.arange(1, x.size + 1)
    s1 = np.cumsum(x)
    s2 = np.cumsum(x**2)
    mean = s1 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1**2 / n) / (n - 1)
    return mean, var
