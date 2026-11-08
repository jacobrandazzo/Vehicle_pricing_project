"""Record baseline results for three fixed vehicle descriptions.

    python make_baseline.py            # full pipeline (calls Claude) -> baseline.txt,
                                       # then the offline run -> baseline_offline.txt
    python make_baseline.py --offline  # only baseline_offline.txt (no API key needed)

Online: each description goes through agent.answer(), exactly as main.py does.
This script wraps two functions only while it runs: agent.run_tool, to record
the arguments Claude sends, and tools.fit_model, to keep the fitted model so the
lambdas can be written out. Neither changes behaviour. Prices come from
predict_price(), which returns them unrounded.

Offline: the parsed queries in OFFLINE_QUERIES go straight into load_data(),
filter_listings(), fit_model() and predict_price(), with no call to Claude. They should match what
Claude parsed in the online run; the online run flags any difference.
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
    """Send each description through the agent and record what it did."""
    import agent  # imported here so --offline never needs an API key
    import tools

    tool_calls = []  # (name, input, fit or None) for every tool Claude asked for
    fits = []  # every model fitted while a tool ran

    original_run_tool = agent.run_tool
    original_fit = tools.fit_model

    def recording_run_tool(name, tool_input):
        fits.clear()
        result = original_run_tool(name, tool_input)
        tool_calls.append((name, dict(tool_input), fits[-1] if fits else None))
        return result

    def recording_fit(*args, **kwargs):
        fit = original_fit(*args, **kwargs)
        fits.append(fit)
        return fit

    agent.run_tool = recording_run_tool
    tools.fit_model = recording_fit

    blocks = [f"Baseline: full pipeline (Claude model {agent.MODEL})"]
    try:
        for description in DESCRIPTIONS:
            tool_calls.clear()
            agent.answer(description, history=[])  # fresh conversation each time

            estimates = [
                (tools._build_query(tool_input), fit)
                for name, tool_input, fit in tool_calls
                if name == "estimate_price" and fit is not None
            ]

            if not estimates:
                blocks.append(
                    f"input: {description}\n"
                    "NO PRICE ESTIMATE: Claude's tool calls were "
                    f"{[(name, tool_input) for name, tool_input, _ in tool_calls]}"
                )
                continue

            for query, fit in estimates:
                block = format_result(description, query, len(tools.vehicles), fit)
                if query != OFFLINE_QUERIES[description]:
                    block += (
                        "\nNOTE: differs from the offline query "
                        f"{OFFLINE_QUERIES[description].model_dump(exclude_none=True)}"
                    )
                blocks.append(block)
    finally:
        agent.run_tool = original_run_tool
        tools.fit_model = original_fit

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
