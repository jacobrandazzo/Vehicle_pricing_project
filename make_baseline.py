"""Record baseline results for three fixed vehicle descriptions.

    python make_baseline.py            # full pipeline (calls Claude) -> baseline.txt,
                                       # then the offline run -> baseline_offline.txt
    python make_baseline.py --offline  # only baseline_offline.txt (no API key needed)

Online: each description goes through the same steps as main.py:
parse_description() (calls Claude), then filter_listings(), fit_model() and
predict_price(). Prices are written unrounded.

Offline: the parsed queries in OFFLINE_QUERIES skip parse_description() and go
straight into filter_listings(), fit_model() and predict_price(), with no call
to Claude. They should match what Claude parsed in the online run; the online
run flags any difference.
"""

import sys
from pathlib import Path

from vehicle_pricing.data import filter_listings, load_data
from vehicle_pricing.model import fit_model, predict_price
from vehicle_pricing.schemas import VehicleQuery

ROOT = Path(__file__).parent
ONLINE_FILE = ROOT / "baseline.txt"
OFFLINE_FILE = ROOT / "baseline_offline.txt"

DESCRIPTIONS = [
    "2015 Camry, 90,000 miles",
    "1996-2000 Honda Civic 200,000 miles",
    "2016 Toyota Tacoma, 120,000 miles",
]

# The parsed query for each description, used by the offline run.
OFFLINE_QUERIES = {
    "2015 Camry, 90,000 miles": VehicleQuery(
        manufacturer="toyota", model="camry", year_min=2015, year_max=2015, odometer=90000
    ),
    "1996-2000 Honda Civic 200,000 miles": VehicleQuery(
        manufacturer="honda", model="civic", year_min=1996, year_max=2000, odometer=200000
    ),
    "2016 Toyota Tacoma, 120,000 miles": VehicleQuery(
        manufacturer="toyota", model="tacoma", year_min=2016, year_max=2016, odometer=120000
    ),
}


def format_result(description, query, rows_after_cleaning, fit):
    """One block of the baseline file. Floats use repr(), so they're unrounded."""
    price = predict_price(fit, query)
    estimate, lower, upper = price.predicted_price, price.lower_bound, price.upper_bound
    return "\n".join([
        f"input: {description}",
        f"parsed query: {query.model_dump(exclude_none=True)}",
        f"rows after cleaning: {rows_after_cleaning}",
        f"listings matched: {price.n_listings}",
        f"box-cox lambda_x (odometer): {fit.lambda_x!r}",
        f"box-cox lambda_y (price): {fit.lambda_y!r}",
        f"predicted price: {estimate!r}",
        f"lower bound: {lower!r}",
        f"upper bound: {upper!r}",
    ])


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
        fit = fit_model(filter_listings(vehicles, query))
        block = format_result(description, query, len(vehicles), fit)
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
        subset = filter_listings(vehicles, query)
        fit = fit_model(subset)
        blocks.append(format_result(description, query, len(vehicles), fit))

    OFFLINE_FILE.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote {OFFLINE_FILE}")


if __name__ == "__main__":
    if "--offline" not in sys.argv:
        run_online()
    run_offline()
