"""Describe a vehicle, get a price estimate and see the plots. Follow-up questions work."""

import matplotlib.pyplot as plt

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import check_manufacturer, comparison_years, filter_listings, load_data
from vehicle_pricing.model import depreciation_curves, drop_outliers, fit_model, predict_price
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
            check_manufacturer(vehicles, query)
            matched = filter_listings(vehicles, query)
            listings = drop_outliers(matched)  # fits once to find prices far off the model
            model = fit_model(listings)
            result = predict_price(model, query) if query.odometer is not None else None
            curves = depreciation_curves(model, query)
        except ParseError as e:
            print(f"Sorry, I couldn't use that description. {e}\n")
            continue
        except ValueError as e:  # unknown make, or too few matching listings
            print(f"{e}\n")  # `previous` is kept, so the user can rephrase
            continue
        previous = query
        years = describe_years(*comparison_years(query))
        dropped = len(matched) - len(listings)
        plural = "" if dropped == 1 else "s"
        dropped_note = f" (dropped {dropped:,} unusual price{plural})" if dropped else ""

        if result is None:
            print(
                "No mileage given, so no price estimate. "
                f"Showing {model.n:,} listings ({years}){dropped_note}."
            )
        else:
            print(
                f"Estimated price: ${result.predicted_price:,.0f} "
                f"(95% range ${result.lower_bound:,.0f} to ${result.upper_bound:,.0f}), "
                f"based on {result.n_listings:,} listings ({years}){dropped_note}."
            )
            if result.extrapolated:
                print("Warning: that mileage is unusual for a car of this age, so the "
                      "estimate leans on other model years and is less reliable.")

        if not curves.age_effect_clear:
            print(f"Note: these listings show no clear age effect (t = {curves.age_t_stat:.2f}), "
                  "so the age curve is only a rough guide.")

        title = " ".join(v for v in (query.manufacturer, query.model) if v).title()
        figures = [
            plot_depreciation(listings, curves, title=title),
            plot_residuals(listings, model, curves, title=f"{title}: residuals vs. odometer"),
        ]
        for fig in figures:
            plt.figure(fig)  # hand the figure to pyplot so plt.show() opens a window for it
        plt.show()  # opens both windows; returns when they are closed
        for fig in figures:
            plt.close(fig)
        print()


if __name__ == "__main__":
    main()
