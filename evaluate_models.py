"""Evaluate the model versions by 5-fold cross-validation, for six popular cars.

    python evaluate_models.py   # takes a few minutes; no API key needed

For each car it takes the 2010-2020 listings, predicts every one with a model
fitted on the other folds (see vehicle_pricing.evaluation), and writes
docs/evaluation.csv (every car x variant x set, all measures) and
docs/evaluation.md (the tables).
"""

import time
from pathlib import Path

import pandas as pd

from vehicle_pricing.data import filter_listings, load_data
from vehicle_pricing.evaluation import VARIANTS, cross_validate, summarize
from vehicle_pricing.model import drop_outliers
from vehicle_pricing.schemas import VehicleQuery

DOCS = Path(__file__).parent / "docs"
CARS = [
    ("toyota", "camry"),
    ("honda", "accord"),
    ("honda", "civic"),
    ("toyota", "tacoma"),
    ("ford", "f-150"),
    ("chevrolet", "silverado 1500"),
]
YEAR_MIN, YEAR_MAX = 2010, 2020
K, SEED = 5, 0
SETS = ("all", "plausible")

NOTE = f"""\
# Model evaluation: {K}-fold cross-validation

Each listing is predicted by a model fitted on the other {K - 1} folds, so no
listing is ever predicted by a model that saw it. All variants use the same
folds (seed {SEED}). Listings are {YEAR_MIN}-{YEAR_MAX} model years.

Each cell is **median % error / coverage**:

- **Median % error**: the middle value of |predicted - actual| / actual, as a
  percent. Lower is better. It is a median, so a few absurd listings don't
  dominate it.
- **Coverage**: the share of actual prices that fall inside the model's 95%
  prediction interval. It should be about 95%. Lower means the intervals are
  too narrow (overconfident); higher means they are wider than they need to be.
- **Average**: the plain mean of the six cars' values.

Outliers were removed from the **training folds only**. Test listings are never
dropped or changed. The two sets differ only in which test listings are scored:

- **all**: every listing, including bad ones such as $500 down-payment ads.
- **plausible**: only the listings kept when `drop_outliers` is run on the whole
  car's listings (3 passes, no features). That filter chooses which rows are
  scored; it is not used for training. It did see every listing, so the
  plausible scores are somewhat flattering to every variant.

"Week 3 style" fits one mileage-only model per model year, so a listing whose
model year has fewer than 30 training listings gets no prediction and is left
out of that variant's scores (counts are listed below the tables).
"""


def main():
    vehicles = load_data()
    rows = []
    sizes = {}
    total = len(CARS) * len(VARIANTS)
    done = 0
    start = time.perf_counter()
    for make, model in CARS:
        car = f"{make} {model}"
        query = VehicleQuery(manufacturer=make, model=model, year_min=YEAR_MIN, year_max=YEAR_MAX)
        listings = filter_listings(vehicles, query)
        plausible = pd.Series(
            listings.index.isin(drop_outliers(listings, max_passes=3).index),
            index=listings.index,
        )
        sizes[car] = (len(listings), int(plausible.sum()))
        print(f"{car}: {len(listings):,} listings, {plausible.sum():,} plausible", flush=True)
        for variant in VARIANTS:
            began = time.perf_counter()
            summary = summarize(cross_validate(listings, variant, k=K, seed=SEED), plausible)
            summary = summary.rename_axis("set").reset_index()
            summary.insert(0, "variant", variant.name)
            summary.insert(0, "car", car)
            rows.append(summary)
            done += 1
            print(f"  [{done}/{total}] {variant.name}: {time.perf_counter() - began:.0f}s "
                  f"(elapsed {(time.perf_counter() - start) / 60:.1f} min)", flush=True)

    results = pd.concat(rows, ignore_index=True)
    DOCS.mkdir(exist_ok=True)
    results.to_csv(DOCS / "evaluation.csv", index=False)
    (DOCS / "evaluation.md").write_text(_markdown(results, sizes), encoding="utf-8")
    print(f"Wrote {DOCS / 'evaluation.csv'} and {DOCS / 'evaluation.md'}")


def _markdown(results: pd.DataFrame, sizes: dict) -> str:
    cars = [f"{make} {model}" for make, model in CARS]
    labels = [model.title() for _, model in CARS]
    parts = [NOTE]
    for name in SETS:
        subset = results[results["set"] == name]
        header = ["Variant", *labels, "Average"]
        lines = [f"## {name.capitalize()} listings", "",
                 "| " + " | ".join(header) + " |",
                 "|" + "---|" * len(header)]
        for variant in VARIANTS:
            by_car = subset[subset["variant"] == variant.name].set_index("car")
            cells = [_cell(by_car.loc[car, "median_abs_pct_error"], by_car.loc[car, "coverage"])
                     for car in cars]
            average = _cell(by_car["median_abs_pct_error"].mean(), by_car["coverage"].mean())
            lines.append("| " + " | ".join([variant.name, *cells, average]) + " |")
        parts.append("\n".join(lines) + "\n")

    counts = ["## Listings scored", "",
              "| Car | All listings | Plausible listings |", "|---|---|---|"]
    counts += [f"| {model.title()} | {sizes[car][0]:,} | {sizes[car][1]:,} |"
               for car, (_, model) in zip(cars, CARS)]
    parts.append("\n".join(counts) + "\n")

    everything = results[results["set"] == "all"]
    missed = everything[everything["n_predicted"] < everything["n_listings"]]
    if missed.empty:
        parts.append("Every variant predicted every listing.\n")
    else:
        lines = ["## Listings that could not be predicted", ""]
        for variant_name, group in missed.groupby("variant", sort=False):
            detail = ", ".join(
                f"{row.car.split(' ', 1)[1].title()} {int(row.n_listings - row.n_predicted):,} "
                f"of {int(row.n_listings):,}" for row in group.itertuples()
            )
            lines.append(f"- {variant_name}: {detail}")
        parts.append("\n".join(lines) + "\n")
    return "\n".join(parts)


def _cell(pct_error: float, coverage: float) -> str:
    if pd.isna(pct_error):
        return "n/a"
    return f"{pct_error:.1f}% / {coverage:.1%}"


if __name__ == "__main__":
    main()
