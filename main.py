"""Describe a vehicle, get a price estimate and see the plots. Follow-up questions work."""

import matplotlib.pyplot as plt

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import comparison_years, load_data
from vehicle_pricing.pipeline import estimate, warning_message
from vehicle_pricing.plots import plot_depreciation, plot_residuals

QUIT_WORDS = ("quit", "exit", "done", "stop")


def describe_years(low, high):
    """The comparison window as words, e.g. '2010–2020 model years'."""
    if low is None and high is None:
        return "all model years"
    if low is None:
        return f"{high} and earlier model years"
    if high is None:
        return f"{low} and later model years"
    if low == high:
        return f"{low} model year"
    return f"{low}–{high} model years"


def main():
    print("Loading listings...")
    vehicles = load_data()
    print(
        "Describe a vehicle, e.g. '2015 Toyota Camry, 90,000 miles'. Follow-ups like "
        "'what about 120k miles?' change the last vehicle. Close the plot windows to "
        "ask again; type 'quit' to exit."
    )

    previous = None  # last successful query, so follow-up questions can change it
    while True:
        text = input("> ").strip()
        if text.lower() in QUIT_WORDS:
            break
        if not text:
            continue

        try:
            query = parse_description(text, previous)
            est = estimate(vehicles, query)
        except ParseError as e:
            print(f"Sorry, I couldn't use that description. {e}\n")
            continue
        except ValueError as e:  # unknown make, or too few matching listings
            print(f"{e}\n")  # `previous` is kept, so the user can rephrase
            continue
        previous = query
        years = describe_years(*comparison_years(query))
        dropped = est.n_dropped
        plural = "" if dropped == 1 else "s"
        dropped_note = f" (dropped {dropped:,} unusual price{plural})" if dropped else ""

        result = est.result
        if result is None:
            print(
                "No mileage given, so no price estimate. "
                f"Showing {est.model.n:,} listings ({years}){dropped_note}."
            )
        else:
            print(
                f"Estimated price: ${result.predicted_price:,.0f} "
                f"(95% range ${result.lower_bound:,.0f} to ${result.upper_bound:,.0f}), "
                f"based on {result.n_listings:,} listings ({years}){dropped_note}."
            )

        for code in est.warnings:  # already in display order
            print(warning_message(code, est))

        title = " ".join(v for v in (query.manufacturer, query.model) if v).title()
        figures = [
            plot_depreciation(est.listings, est.curves, title=title),
            plot_residuals(est.listings, est.model, est.curves,
                           title=f"{title}: residuals vs. odometer"),
        ]
        for fig in figures:
            plt.figure(fig)  # hand the figure to pyplot so plt.show() opens a window for it
        plt.show()  # opens both windows; returns when they are closed
        for fig in figures:
            plt.close(fig)
        print()


if __name__ == "__main__":
    main()
