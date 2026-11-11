"""Screen candidate extra terms (mileage x age, mileage^2, age^2) with 5-fold CV.

    python experiments/screen_extra_terms.py   # under a minute; no API key needed

Not part of the package. For each car (the six in evaluate_models.py,
2010-2020 listings) it runs the default pipeline once on all the listings
(drop_outliers, then fit_model, both with DEFAULT_FEATURES) and keeps that
fit's three Box-Cox lambdas fixed. Then, on the transformed variables, it
compares linear models with and without each extra term by 5-fold CV, refitting
only the coefficients (np.linalg.lstsq) on each training fold.

Because the lambdas and the outlier removal used all of each car's listings,
this is a screen, not a full evaluation. Writes docs/feature_screen.md.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import boxcox

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # so evaluate_models (at the project root) can be imported

from evaluate_models import CARS, K, SEED, YEAR_MAX, YEAR_MIN  # noqa: E402
from vehicle_pricing.data import filter_listings, load_data  # noqa: E402
from vehicle_pricing.evaluation import assign_folds  # noqa: E402
from vehicle_pricing.model import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_FEATURES,
    _build_design,
    _half_width,
    boxcox_inverse,
    drop_outliers,
    fit_model,
    listing_features,
)
from vehicle_pricing.schemas import VehicleQuery  # noqa: E402

OUTPUT = ROOT / "docs" / "feature_screen.md"
BASE = "Base (the model's own columns)"
VARIANTS = [BASE, "+ miles × age", "+ miles²", "+ age²"]


def main():
    vehicles = load_data()
    rows = []
    for make, model_name in CARS:
        query = VehicleQuery(manufacturer=make, model=model_name, year_min=YEAR_MIN, year_max=YEAR_MAX)
        listings = drop_outliers(filter_listings(vehicles, query), features=DEFAULT_FEATURES)
        model = fit_model(listings, features=DEFAULT_FEATURES)
        base = _build_design(model, listings["odometer"], listings["age"],
                             listing_features(model, listings))
        ty = boxcox(listings["price"].to_numpy(dtype=float), model.lambda_y)
        miles, age = _standardize(base[:, 1]), _standardize(base[:, 2])
        designs = {
            BASE: base,
            "+ miles × age": np.column_stack([base, miles * age]),
            "+ miles²": np.column_stack([base, miles ** 2]),
            "+ age²": np.column_stack([base, age ** 2]),
        }
        folds = assign_folds(len(listings), K, SEED)
        for variant, X in designs.items():
            error, inside = _cross_validate(X, ty, listings["price"].to_numpy(dtype=float),
                                            folds, model.lambda_y)
            rows.append({"car": model_name.title(), "variant": variant, "features": model.features,
                         "n": len(listings), "median_abs_pct_error": np.median(error),
                         "coverage": inside.mean()})
        print(f"{make} {model_name}: {len(listings):,} listings, features {model.features}",
              flush=True)

    results = pd.DataFrame(rows)
    OUTPUT.write_text(_markdown(results), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


def _standardize(values: np.ndarray) -> np.ndarray:
    """Centre and scale. The fit doesn't depend on these constants (the base design
    already has an intercept and both linear terms); it just keeps the numbers tame."""
    return (values - values.mean()) / values.std()


def _cross_validate(X, ty, price, folds, lambda_y):
    """Refit the coefficients on each training fold; return |% error| and inside-interval flags."""
    error = np.empty(len(ty))
    inside = np.empty(len(ty), dtype=bool)
    for fold in range(K):
        test = folds == fold
        X_train, ty_train = X[~test], ty[~test]
        coef, *_ = np.linalg.lstsq(X_train, ty_train, rcond=None)
        dof = len(ty_train) - X.shape[1]
        s = np.sqrt(np.sum((ty_train - X_train @ coef) ** 2) / dof)
        r_inv = np.linalg.inv(np.linalg.qr(X_train, mode="r"))  # (X^T X)^-1 = R^-1 R^-T
        y_hat = X[test] @ coef
        half = _half_width(X[test], r_inv @ r_inv.T, s, dof, DEFAULT_ALPHA)  # t interval, leverage
        estimate = boxcox_inverse(y_hat, lambda_y)
        lower = boxcox_inverse(y_hat - half, lambda_y)
        upper = boxcox_inverse(y_hat + half, lambda_y)
        error[test] = np.abs(estimate - price[test]) / price[test] * 100
        inside[test] = (price[test] >= lower) & (price[test] <= upper)
    return error, inside


def _markdown(results: pd.DataFrame) -> str:
    cars = list(dict.fromkeys(results["car"]))
    err = results.pivot(index="variant", columns="car", values="median_abs_pct_error")
    cov = results.pivot(index="variant", columns="car", values="coverage")
    features = results.drop_duplicates("car").set_index("car")["features"]

    lines = [
        "# Screen of extra terms: 5-fold cross-validation",
        "",
        "Does adding a mileage × age interaction, or a squared term for mileage or",
        "age, improve the final model? Generated by `experiments/screen_extra_terms.py`.",
        "",
        "**Method.** For each car (2010–2020 listings) I ran the default pipeline once",
        "on all of its listings: `drop_outliers`, then `fit_model`, both with the",
        "default features. I kept that fit's three Box-Cox λs **fixed**, and on the",
        "transformed variables compared linear models with and without each extra term",
        f"by {K}-fold cross-validation (seed {SEED}, the same `assign_folds` as",
        "`evaluation.py`), refitting only the coefficients on each training fold. The",
        "extra terms are built from the standardized transformed mileage and age.",
        "Each listing's prediction uses its own values, with a t interval that",
        "includes leverage, as in `model.py`.",
        "",
        "**This is a screen, not a full evaluation.** The λs and the outlier removal",
        "were fitted on all of each car's listings, including the ones later used as",
        "test folds, and only listings kept by `drop_outliers` are scored. So the",
        "numbers are not comparable with `docs/evaluation.md`; only the differences",
        "between rows here are meaningful.",
        "",
        "## Median % error / 95% coverage",
        "",
        "| Variant | " + " | ".join(cars) + " | Average |",
        "|" + "---|" * (len(cars) + 2),
    ]
    for variant in VARIANTS:
        cells = [f"{err.loc[variant, car]:.2f}% / {cov.loc[variant, car]:.1%}" for car in cars]
        average = f"{err.loc[variant, cars].mean():.2f}% / {cov.loc[variant, cars].mean():.1%}"
        lines.append(f"| {variant} | " + " | ".join(cells) + f" | {average} |")

    lines += [
        "",
        "## Change in median % error vs base (percentage points; negative = better)",
        "",
        "| Variant | " + " | ".join(cars) + " | Average |",
        "|" + "---|" * (len(cars) + 2),
    ]
    for variant in VARIANTS[1:]:
        delta = err.loc[variant, cars] - err.loc[BASE, cars]
        lines.append(f"| {variant} | " + " | ".join(f"{d:+.2f}" for d in delta)
                     + f" | {delta.mean():+.2f} |")

    lines += ["", "Base columns per car: intercept, mileage, age, plus "
              + "; ".join(f"{car}: {', '.join(features[car]) or 'no features'}" for car in cars)
              + ".", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
