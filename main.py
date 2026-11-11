"""Describe a vehicle, get a price estimate and see the plots. Follow-up questions work."""

import matplotlib.pyplot as plt

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import check_manufacturer, comparison_years, filter_listings, load_data
from vehicle_pricing.model import fit_model, predict_price
from vehicle_pricing.plots import plot_fit, plot_residuals

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
            listings = filter_listings(vehicles, query)
            model = fit_model(listings)
            result = predict_price(model, query) if query.odometer is not None else None
        except ParseError as e:
            print(f"Sorry, I couldn't use that description. {e}\n")
            continue
        except ValueError as e:  # unknown make, or too few matching listings
            print(f"{e}\n")  # `previous` is kept, so the user can rephrase
            continue
        previous = query
        years = describe_years(*comparison_years(query))

        if result is None:
            print(
                "No mileage given, so no price estimate. "
                f"Showing {model.n:,} listings ({years})."
            )
        else:
            print(
                f"Estimated price: ${result.predicted_price:,.0f} "
                f"(95% range ${result.lower_bound:,.0f} to ${result.upper_bound:,.0f}), "
                f"based on {result.n_listings:,} listings ({years})."
            )
            if result.extrapolated:
                print("Warning: that mileage is outside the range of the listings, "
                      "so this estimate is less reliable.")

        title = " ".join(v for v in (query.manufacturer, query.model) if v).title()
        figures = [
            plot_fit(listings, model, result, query=query, title=title),
            plot_residuals(listings, model, title=f"{title}: residuals vs. odometer"),
        ]
        plt.show()  # opens both windows; returns when they are closed
        for fig in figures:
            plt.close(fig)
        print()


if __name__ == "__main__":
    main()
