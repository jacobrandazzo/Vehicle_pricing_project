"""Terminal app: describe a vehicle, get a price estimate and see the plots."""

import matplotlib.pyplot as plt

from vehicle_pricing.agent import ParseError, parse_description
from vehicle_pricing.data import check_manufacturer, filter_listings, load_data
from vehicle_pricing.model import fit_model, predict_price
from vehicle_pricing.plots import plot_fit, plot_residuals

QUIT_WORDS = ("quit", "exit", "done", "stop")


def main():
    print("Loading listings...")
    vehicles = load_data()
    print(
        "Vehicle price estimator. Describe a vehicle, e.g. '2015 Toyota Camry, 90,000 miles'. "
        "Include the mileage for a price estimate. Follow-ups like 'what about 120k miles?' "
        "change the last vehicle. Close the plot windows to ask again; type 'quit' to exit."
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
        except (ParseError, ValueError) as e:  # ValueError: unknown make, too few listings
            print(e)  # `previous` is kept, so the user can rephrase the follow-up
            print()
            continue
        previous = query

        title = " ".join(v for v in (query.manufacturer, query.model) if v).title()
        if result is None:
            print(f"No mileage given, so no price estimate. Showing {model.n:,} listings.")
        else:
            print(
                f"Estimated price: ${result.predicted_price:,.0f} "
                f"(95% range ${result.lower_bound:,.0f} to ${result.upper_bound:,.0f}), "
                f"based on {result.n_listings:,} listings."
            )
            if result.extrapolated:
                print("Warning: that mileage is outside the range of the listings, "
                      "so this estimate is less reliable.")

        figures = [
            plot_fit(listings, model, result, title=title),
            plot_residuals(listings, model, title=f"{title}: residuals vs. odometer"),
        ]
        plt.show()  # opens both windows; returns when they are closed
        for fig in figures:
            plt.close(fig)
        print()


if __name__ == "__main__":
    main()
