"""Record baseline results for four fixed vehicle descriptions.

    python make_baseline.py            # full pipeline (calls Claude) -> baseline.txt,
                                       # then the offline run -> baseline_offline.txt
    python make_baseline.py --offline  # only baseline_offline.txt (no API key needed)

Online: each description goes through the same steps as main.py:
parse_description() (calls Claude), then filter_listings(), drop_outliers(),
fit_model() and predict_price(). Prices are written unrounded.

Offline: the parsed queries in OFFLINE_QUERIES skip parse_description() and go
straight into filter_listings(), drop_outliers(), fit_model() and
predict_price(), with no call to Claude. They should match what Claude parsed in the online run; the online
run flags any difference.
"""

import sys
from pathlib import Path

from vehicle_pricing.data import comparison_years, filter_listings, load_data
from vehicle_pricing.model import (
    car_model_year,
    depreciation_curves,
    drop_outliers,
    fit_model,
    predict_price,
    query_age,
    typical_mileage_range,
)
from vehicle_pricing.schemas import VehicleQuery

ROOT = Path(__file__).parent
ONLINE_FILE = ROOT / "baseline.txt"
OFFLINE_FILE = ROOT / "baseline_offline.txt"

DESCRIPTIONS = [
    "2015 Camry, 90,000 miles",
    "1996-2000 Honda Civic 200,000 miles",
    "2016 Toyota Tacoma, 120,000 miles",
    "2015 Honda Accord",  # no mileage: the model is fitted, but there's no price
]

# The parsed query for each description, used by the offline run.
OFFLINE_QUERIES = {
    "2015 Camry, 90,000 miles": VehicleQuery(
        manufacturer="toyota", model="camry", year=2015, odometer=90000
    ),
    "1996-2000 Honda Civic 200,000 miles": VehicleQuery(
        manufacturer="honda", model="civic", year_min=1996, year_max=2000, odometer=200000
    ),
    "2016 Toyota Tacoma, 120,000 miles": VehicleQuery(
        manufacturer="toyota", model="tacoma", year=2016, odometer=120000
    ),
    "2015 Honda Accord": VehicleQuery(
        manufacturer="honda", model="accord", year=2015
    ),
}


def fit_listings(vehicles, query):
    """Filter, drop outlier prices, and fit: the same pipeline as main.py.

    Returns (listings matched, outliers dropped, fitted model).
    """
    matched = filter_listings(vehicles, query)
    kept = drop_outliers(matched)
    return len(matched), len(matched) - len(kept), fit_model(kept)


def format_result(description, query, rows_after_cleaning, matched, dropped, fit):
    """One block of the baseline file. Floats use repr(), so they're unrounded.

    Without a mileage there's no price, so only the fit is recorded (main.py
    likewise shows only the plots in that case).
    """
    curves = depreciation_curves(fit, query)
    lines = [
        f"input: {description}",
        f"parsed query: {query.model_dump(exclude_none=True)}",
        f"comparison years: {comparison_years(query)}",
        f"rows after cleaning: {rows_after_cleaning}",
        f"listings matched: {matched}",
        f"outliers dropped: {dropped}",
        f"box-cox lambda_x (odometer): {fit.lambda_x!r}",
        f"box-cox lambda_age (age + 1): {fit.lambda_age!r}",
        f"box-cox lambda_y (price): {fit.lambda_y!r}",
        f"coefficients b0, b1 (odometer), b2 (age): {', '.join(repr(c) for c in fit.coef)}",
        f"r-squared (transformed): {fit.r2!r}",
        f"car's age used (years): {query_age(fit, query)!r}",
        f"age effect t-stat: {curves.age_t_stat!r}",
        f"age effect clear: {curves.age_effect_clear}",
        "typical mileage for this model year: "
        f"{typical_mileage_range(fit, car_model_year(fit, query))!r}",
    ]
    if query.odometer is None:
        lines += [
            "predicted price: none (no mileage given)",
            "extrapolated: n/a (no mileage given)",
        ]
    else:
        price = predict_price(fit, query)
        lines += [
            f"predicted price: {price.predicted_price!r}",
            f"lower bound: {price.lower_bound!r}",
            f"upper bound: {price.upper_bound!r}",
            f"extrapolated: {price.extrapolated}",
        ]
    return "\n".join(lines)


def run_online():
    """Run each description through the full pipeline, including Claude."""
    from vehicle_pricing.agent import ParseError, parse_description  # needs an API key

    vehicles = load_data()
    blocks = ["Baseline: full pipeline (parse_description via Claude)"]
    for description in DESCRIPTIONS:
        try:
            query = parse_description(description)
        except ParseError as e:
            blocks.append(f"input: {description}\nPARSE ERROR: {e}")
            continue
        matched, dropped, fit = fit_listings(vehicles, query)
        block = format_result(description, query, len(vehicles), matched, dropped, fit)
        if query != OFFLINE_QUERIES[description]:
            block += (
                "\nNOTE: differs from the offline query "
                f"{OFFLINE_QUERIES[description].model_dump(exclude_none=True)}"
            )
        blocks.append(block)

    ONLINE_FILE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {ONLINE_FILE}")


def run_offline():
    """Run the fixed parsed queries through filtering and the model, without Claude."""
    vehicles = load_data()
    blocks = ["Baseline: offline (no Claude; fixed parsed queries)"]
    for description in DESCRIPTIONS:
        query = OFFLINE_QUERIES[description]
        matched, dropped, fit = fit_listings(vehicles, query)
        blocks.append(format_result(description, query, len(vehicles), matched, dropped, fit))

    OFFLINE_FILE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {OFFLINE_FILE}")


if __name__ == "__main__":
    if "--offline" not in sys.argv:
        run_online()
    run_offline()
