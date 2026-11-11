"""Compare versions of the price model by k-fold cross-validation.

Every listing gets a prediction from a model that never saw it: the listings
are split into k folds, and each fold is predicted by a model fitted on the
other k - 1 folds. Everything that learns from data (outlier removal, the
lambda search, the coefficients) happens on the training folds only. Test
listings are never dropped or changed, and each is predicted at its own
mileage, age and feature values.

A Variant describes one model version, and VARIANTS lists the ones compared.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from scipy.special import boxcox

# Private helpers from model.py, reused so the two-coefficient Week 3 model
# computes its interval and its fits the same way as the full model.
from vehicle_pricing.model import (
    DEFAULT_ALPHA,
    _half_width,
    _rss,
    boxcox_inverse,
    drop_outliers,
    fit_model,
    listing_features,
    predict_interval,
)
from vehicle_pricing.schemas import MIN_LISTINGS

LAMBDA_RANGE = (-2.0, 2.0)  # the same search as fit_model's defaults
GRID_STEP = 0.25
MEASURES = [
    "n_listings", "n_predicted", "median_abs_error", "median_abs_pct_error",
    "coverage", "median_interval_width",
]


@dataclass(frozen=True)
class Variant:
    """One version of the model to evaluate.

    features: optional 0/1 predictors from model.FEATURES.
    outlier_passes: passes of drop_outliers on the training folds (0 = none).
    per_year_mileage_only: the Week 3 model instead: for each model year, a
        mileage-only Box-Cox model fitted on training listings of that year.
        It has no features and no outlier removal.
    """

    name: str
    features: tuple[str, ...] = ()
    outlier_passes: int = 0
    per_year_mileage_only: bool = False

    def __post_init__(self):
        if self.per_year_mileage_only and (self.features or self.outlier_passes):
            raise ValueError("The Week 3 style model has no features or outlier removal.")


VARIANTS = [
    Variant("Week 3 style (mileage only, own year)", per_year_mileage_only=True),
    Variant("Mileage + age, no outlier removal"),
    Variant("Mileage + age, 1 pass", outlier_passes=1),
    Variant("Mileage + age, 3 passes", outlier_passes=3),
    Variant("+ clean title", features=("clean_title",), outlier_passes=3),
    Variant("+ clean title + 4WD", features=("clean_title", "four_wd"), outlier_passes=3),
]


def assign_folds(n: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """A fold number (0 to k - 1) for each of n listings; fold sizes differ by at most 1.

    It depends only on n, k and seed, so every variant evaluated on the same
    listings gets identical folds.
    """
    if not 2 <= k <= n:
        raise ValueError(f"Need 2 <= k <= number of listings, got k={k} for {n} listings")
    order = np.random.default_rng(seed).permutation(n)
    folds = np.empty(n, dtype=int)
    folds[order] = np.arange(n) % k
    return folds


def cross_validate(
    listings: pd.DataFrame,
    variant: Variant,
    *,
    k: int = 5,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Predict every listing with a model fitted on the other folds.

    Returns one row per listing, in the listings' own index: its `fold`, the
    actual `price`, and the `predicted` median with its `lower` and `upper`
    1 - alpha prediction bounds (all in dollars). Predictions are empty (NaN)
    where the variant can't make one (Week 3 style, for a model year with
    fewer than MIN_LISTINGS training listings).

    Per fold, outlier removal, the lambda search and the coefficients use the
    training folds only. Test listings are never dropped or changed.
    """
    folds = assign_folds(len(listings), k, seed)
    predicted, lower, upper = (np.full(len(listings), np.nan) for _ in range(3))
    for fold in range(k):
        is_test = folds == fold
        train, test = listings[~is_test], listings[is_test]
        if variant.per_year_mileage_only:
            estimates = _predict_per_year(train, test, alpha)
        else:
            estimates = _predict_full(train, test, variant, alpha)
        predicted[is_test], lower[is_test], upper[is_test] = estimates
    return pd.DataFrame(
        {"fold": folds, "price": listings["price"].to_numpy(dtype=float),
         "predicted": predicted, "lower": lower, "upper": upper},
        index=listings.index,
    )


def summarize(results: pd.DataFrame, plausible_mask: pd.Series) -> pd.DataFrame:
    """Accuracy measures for all the results, and for the plausible listings only.

    plausible_mask is a True/False Series on the results' index. Returns one
    row per set ("all", "plausible") with: n_listings, n_predicted (those
    with a prediction), median_abs_error ($), median_abs_pct_error (% of the
    actual price), coverage (share of actual prices inside the interval) and
    median_interval_width ($). Measures use predicted listings only.
    """
    plausible = plausible_mask.loc[results.index].to_numpy(dtype=bool)
    return pd.DataFrame(
        {"all": _measures(results), "plausible": _measures(results[plausible])}
    ).T[MEASURES]


def _measures(results: pd.DataFrame) -> pd.Series:
    done = results.dropna(subset=["predicted"])
    if done.empty:
        return pd.Series({m: np.nan for m in MEASURES} | {"n_listings": len(results),
                                                           "n_predicted": 0})
    error = (done["predicted"] - done["price"]).abs()
    inside = (done["price"] >= done["lower"]) & (done["price"] <= done["upper"])
    return pd.Series({
        "n_listings": len(results),
        "n_predicted": len(done),
        "median_abs_error": error.median(),
        "median_abs_pct_error": (error / done["price"] * 100).median(),
        "coverage": inside.mean(),
        "median_interval_width": (done["upper"] - done["lower"]).median(),
    })


def _predict_full(train, test, variant, alpha):
    """Mileage + age (+ features) model: fit on the training folds, predict the test rows."""
    if variant.outlier_passes > 0:
        train = drop_outliers(train, max_passes=variant.outlier_passes, features=variant.features)
    model = fit_model(train, features=variant.features)
    return predict_interval(
        model, test["odometer"], test["age"], alpha,
        feature_values=listing_features(model, test),  # each listing's own values
    )


def _predict_per_year(train, test, alpha):
    """Week 3 style: one mileage-only model per model year, fitted on that year's training listings."""
    predicted, lower, upper = (np.full(len(test), np.nan) for _ in range(3))
    test_years = test["year"].to_numpy()
    train_years = train["year"].to_numpy()
    for year in np.unique(test_years):
        same_year = train[train_years == year]
        if len(same_year) < MIN_LISTINGS:
            continue  # too few listings of that year: leave its predictions empty
        in_year = test_years == year
        fit = _fit_mileage_only(same_year["odometer"], same_year["price"])
        predicted[in_year], lower[in_year], upper[in_year] = _predict_mileage_only(
            fit, test.loc[in_year, "odometer"], alpha
        )
    return predicted, lower, upper


class _MileageOnlyFit(NamedTuple):
    lambda_x: float
    lambda_y: float
    coef: np.ndarray  # intercept, slope (transformed space)
    xtx_inv: np.ndarray
    resid_se: float  # s, with n - 2 degrees of freedom
    n: int


def _fit_mileage_only(odometer: ArrayLike, price: ArrayLike) -> _MileageOnlyFit:
    """boxcox(price, lambda_y) = b0 + b1 * boxcox(odometer, lambda_x), by profile likelihood.

    The lambdas maximise the same profile log-likelihood as fit_model
    (-(n/2) log(RSS / n) + (lambda_y - 1) * sum(log(price))), searched the
    same way: a coarse grid, then Nelder-Mead within the bounds.
    """
    x = np.asarray(odometer, dtype=float)
    y = np.asarray(price, dtype=float)
    n = y.size
    sum_log_y = float(np.sum(np.log(y)))
    grid = np.round(np.arange(LAMBDA_RANGE[0], LAMBDA_RANGE[1] + GRID_STEP / 2, GRID_STEP), 10)
    ty = np.column_stack([boxcox(y, ly) for ly in grid])

    best = (-np.inf, 0.0, 0.0)
    for lx in grid:
        X = np.column_stack([np.ones(n), boxcox(x, lx)])
        ll = -(n / 2) * np.log(_rss(X, ty) / n) + (grid - 1) * sum_log_y  # one per lambda_y
        i = int(np.argmax(ll))
        if ll[i] > best[0]:
            best = (float(ll[i]), float(lx), float(grid[i]))

    def negative_ll(p):
        X = np.column_stack([np.ones(n), boxcox(x, p[0])])
        rss = _rss(X, boxcox(y, p[1])[:, None])[0]
        return -(-(n / 2) * np.log(rss / n) + (p[1] - 1) * sum_log_y)

    refined = minimize(negative_ll, x0=best[1:], method="Nelder-Mead",
                       bounds=[LAMBDA_RANGE] * 2)
    lam_x, lam_y = (float(v) for v in refined.x)

    X = np.column_stack([np.ones(n), boxcox(x, lam_x)])
    ty = boxcox(y, lam_y)
    coef, *_ = np.linalg.lstsq(X, ty, rcond=None)
    rss = float(np.sum((ty - X @ coef) ** 2))
    r_inv = np.linalg.inv(np.linalg.qr(X, mode="r"))  # (X^T X)^-1 = R^-1 R^-T
    return _MileageOnlyFit(lam_x, lam_y, coef, r_inv @ r_inv.T, float(np.sqrt(rss / (n - 2))), n)


def _predict_mileage_only(fit: _MileageOnlyFit, odometer: ArrayLike, alpha: float):
    """(estimate, lower, upper) in dollars; the interval uses n - 2 degrees of freedom."""
    x = np.asarray(odometer, dtype=float)
    x0 = np.column_stack([np.ones(x.size), boxcox(x, fit.lambda_x)])
    y_hat = x0 @ fit.coef
    half = _half_width(x0, fit.xtx_inv, fit.resid_se, fit.n - 2, alpha)
    return (
        boxcox_inverse(y_hat, fit.lambda_y),
        boxcox_inverse(y_hat - half, fit.lambda_y),
        boxcox_inverse(y_hat + half, fit.lambda_y),
    )
